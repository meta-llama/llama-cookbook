/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.llamaandroiddemo;

import java.util.Arrays;
import java.util.List;

public class ModelUtils {
  public static List<String> getSupportedRemoteModels() {
      // UPDATE THIS TO THE RELEVANT MODELS YOU WANT TO USE
      // NOTE THAT SOME PROVIDERS MIGHT HAVE DIFFERENT MODEL NAMING FORMAT
    return Arrays.asList(
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            "meta-llama/Llama-4-Scout-17B-16E-Instruct"
            );
  }
}
