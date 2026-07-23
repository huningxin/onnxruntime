// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { DataType } from '../../../wasm-common';
import { TensorView } from '../../tensor-view';
import { ShapeUtil } from '../../util';
import { AttributeWithCacheKey, createAttributeWithCacheKey } from '../attribute-with-cache-key';
import { ComputeContext, ProgramInfo } from '../types';

import { createTensorShapeVariables, inputVariable, outputVariable, ShaderHelper } from './common';

export interface TriluAttributes extends AttributeWithCacheKey {
  readonly upper: boolean;
}

const createTriluProgramInfo = (inputs: readonly TensorView[], attributes: TriluAttributes): ProgramInfo => {
  const inputShape = inputs[0].dims;
  const inputType = inputs[0].dataType;
  const rank = inputShape.length;
  const outputSize = ShapeUtil.size(inputShape);

  let k = 0;
  if (inputs.length > 1) {
    const kInput = inputs[1];
    if (kInput.dataType === DataType.int32) {
      k = kInput.getInt32Array()[0];
    } else if (kInput.dataType === DataType.int64) {
      k = Number(kInput.getBigInt64Array()[0]);
    } else {
      throw new Error('Trilu input "k" must be int32 or int64.');
    }
  }

  const input = inputVariable('input', inputType, rank);
  const output = outputVariable('output', inputType, rank);
  const condition = attributes.upper ? '(row + uniforms.k) <= col' : '(row + uniforms.k) >= col';

  const getShaderSource = (shaderHelper: ShaderHelper) => `
      ${shaderHelper
        .registerUniform('outputSize', 'u32')
        .registerUniform('k', 'i32')
        .declareVariables(input, output)}
      ${shaderHelper.mainStart()}
        ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes('uniforms.outputSize')}
        let outputIndices = ${output.offsetToIndices('global_idx')};
        let row = i32(${output.indicesGet('outputIndices', rank - 2)});
        let col = i32(${output.indicesGet('outputIndices', rank - 1)});
        let value = ${input.getByOffset('global_idx')};
        ${output.setByOffset('global_idx', `${condition} ? value : ${output.type.value}(0)`)};
      }`;

  return {
    name: 'Trilu',
    shaderCache: { hint: attributes.cacheKey, inputDependencies: ['rank'] },
    getRunData: () => ({
      outputs: [{ dims: inputShape, dataType: inputType }],
      dispatchGroup: { x: Math.ceil(outputSize / 64 /* workgroup size */) },
      programUniforms: [
        { type: DataType.uint32, data: outputSize },
        { type: DataType.int32, data: k },
        ...createTensorShapeVariables(inputShape, inputShape),
      ],
    }),
    getShaderSource,
  };
};

export const trilu = (context: ComputeContext, attributes: TriluAttributes): void => {
  if (context.inputs.length !== 1 && context.inputs.length !== 2) {
    throw new Error('Trilu requires 1 or 2 inputs.');
  }

  if (context.inputs[0].dims.length < 2) {
    throw new Error('Trilu input must have rank >= 2.');
  }

  context.compute(createTriluProgramInfo(context.inputs, attributes), { inputs: [0] });
};

export const parseTriluAttributes = (attributes: Record<string, unknown>): TriluAttributes =>
  createAttributeWithCacheKey({ upper: (attributes.upper as number | undefined) !== 0 });
