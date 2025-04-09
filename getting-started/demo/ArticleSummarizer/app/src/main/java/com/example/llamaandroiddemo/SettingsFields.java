/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.llamaandroiddemo;

public class SettingsFields {

  public String getModelFilePath() {
    return modelFilePath;
  }

  public String getTokenizerFilePath() {
    return tokenizerFilePath;
  }

  public double getTemperature() {
    return temperature;
  }

  public String getSystemPrompt() {
    return systemPrompt;
  }

  public ModelType getModelType() {
    return modelType;
  }

  public BackendType getBackendType() {
    return backendType;
  }

  public String getRemoteURL() {
    return remoteURL;
  }

  public boolean getIsClearChatHistory() {
    return isClearChatHistory;
  }

  public boolean getIsLoadModel() {
    return isLoadModel;
  }

  public String getRemoteModel() {
    return remoteModel;
  }

  private String modelFilePath;
  private String tokenizerFilePath;
  private double temperature;
  private String systemPrompt;
  private boolean isClearChatHistory;
  private boolean isLoadModel;
  private ModelType modelType;
  private BackendType backendType;
  private String remoteURL;
  private String remoteModel;

  public SettingsFields() {
    ModelType DEFAULT_MODEL = ModelType.LLAMA_3;
    BackendType DEFAULT_BACKEND = BackendType.XNNPACK;

    modelFilePath = "";
    tokenizerFilePath = "";
    temperature = SettingsActivity.TEMPERATURE_MIN_VALUE;
    systemPrompt = "";
    isClearChatHistory = false;
    isLoadModel = false;
    modelType = DEFAULT_MODEL;
    backendType = DEFAULT_BACKEND;
    remoteURL = "";
    remoteModel = "";
  }

  public SettingsFields(SettingsFields settingsFields) {
    this.modelFilePath = settingsFields.modelFilePath;
    this.tokenizerFilePath = settingsFields.tokenizerFilePath;
    this.temperature = settingsFields.temperature;
    this.systemPrompt = settingsFields.getSystemPrompt();
    this.isClearChatHistory = settingsFields.getIsClearChatHistory();
    this.isLoadModel = settingsFields.getIsLoadModel();
    this.modelType = settingsFields.modelType;
    this.backendType = settingsFields.backendType;
    this.remoteURL = settingsFields.remoteURL;
    this.remoteModel = settingsFields.remoteModel;
  }

  public void saveModelPath(String modelFilePath) {
    this.modelFilePath = modelFilePath;
  }

  public void saveTokenizerPath(String tokenizerFilePath) {
    this.tokenizerFilePath = tokenizerFilePath;
  }

  public void saveModelType(ModelType modelType) {
    this.modelType = modelType;
  }

  public void saveBackendType(BackendType backendType) {
    this.backendType = backendType;
  }

  public void saveParameters(Double temperature) {
    this.temperature = temperature;
  }

  public void savePrompts(String systemPrompt) {
    this.systemPrompt = systemPrompt;
  }

  public void saveIsClearChatHistory(boolean needToClear) {
    this.isClearChatHistory = needToClear;
  }

  public void saveLoadModelAction(boolean shouldLoadModel) {
    this.isLoadModel = shouldLoadModel;
  }

  public void saveRemoteURL(String url) {
    this.remoteURL = url;
  }

  public void saveRemoteModel(String model) {
    this.remoteModel = model;
  }

  public boolean equals(SettingsFields anotherSettingsFields) {
    if (this == anotherSettingsFields) return true;
    return modelFilePath.equals(anotherSettingsFields.modelFilePath)
                   && tokenizerFilePath.equals(anotherSettingsFields.tokenizerFilePath)
                   && temperature == anotherSettingsFields.temperature
                   && systemPrompt.equals(anotherSettingsFields.systemPrompt)
                   && isClearChatHistory == anotherSettingsFields.isClearChatHistory
                   && isLoadModel == anotherSettingsFields.isLoadModel
                   && modelType == anotherSettingsFields.modelType
                   && backendType == anotherSettingsFields.backendType
                   && remoteURL.equals(anotherSettingsFields.remoteURL)
                   && remoteModel.equals(anotherSettingsFields.remoteModel);
  }
}
