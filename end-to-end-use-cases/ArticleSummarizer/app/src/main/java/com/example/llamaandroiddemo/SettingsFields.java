/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.llamaandroiddemo;

public class SettingsFields {

  public double getTemperature() {
    return temperature;
  }

  public String getSystemPrompt() {
    return systemPrompt;
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

  private double temperature;
  private String systemPrompt;
  private boolean isClearChatHistory;
  private boolean isLoadModel;
  private String remoteURL;
  private String remoteModel;

  public SettingsFields() {
    temperature = SettingsActivity.TEMPERATURE_MIN_VALUE;
    systemPrompt = "";
    isClearChatHistory = false;
    isLoadModel = false;
    remoteURL = "";
    remoteModel = "";
  }

  public SettingsFields(SettingsFields settingsFields) {
    this.temperature = settingsFields.temperature;
    this.systemPrompt = settingsFields.getSystemPrompt();
    this.isClearChatHistory = settingsFields.getIsClearChatHistory();
    this.isLoadModel = settingsFields.getIsLoadModel();
    this.remoteURL = settingsFields.remoteURL;
    this.remoteModel = settingsFields.remoteModel;
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
    return temperature == anotherSettingsFields.temperature
                   && systemPrompt.equals(anotherSettingsFields.systemPrompt)
                   && isClearChatHistory == anotherSettingsFields.isClearChatHistory
                   && isLoadModel == anotherSettingsFields.isLoadModel
                   && remoteURL.equals(anotherSettingsFields.remoteURL)
                   && remoteModel.equals(anotherSettingsFields.remoteModel);
  }
}
