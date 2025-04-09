/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.llamaandroiddemo;

import android.content.Context;
import android.content.SharedPreferences;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import java.lang.reflect.Type;
import java.util.ArrayList;

public class DemoSharedPreferences {
  Context context;
  SharedPreferences sharedPreferences;

  public DemoSharedPreferences(Context context) {
    this.context = context;
    this.sharedPreferences = getSharedPrefs();
  }

  private SharedPreferences getSharedPrefs() {
    return context.getSharedPreferences(
        context.getString(R.string.demo_pref_file_key), Context.MODE_PRIVATE);
  }

  public String getSavedMessages() {
    return sharedPreferences.getString(context.getString(R.string.saved_messages_json_key), "");
  }

  public void addMessages(MessageAdapter messageAdapter) {
    SharedPreferences.Editor editor = sharedPreferences.edit();
    Gson gson = new Gson();
    String msgJSON = gson.toJson(messageAdapter.getSavedMessages());
    editor.putString(context.getString(R.string.saved_messages_json_key), msgJSON);
    editor.apply();
  }

  public void removeExistingMessages() {
    SharedPreferences.Editor editor = sharedPreferences.edit();
    editor.remove(context.getString(R.string.saved_messages_json_key));
    editor.apply();
  }

  public void addSettings(SettingsFields settingsFields) {
    SharedPreferences.Editor editor = sharedPreferences.edit();
    Gson gson = new Gson();
    String settingsJSON = gson.toJson(settingsFields);
    editor.putString(context.getString(R.string.settings_json_key), settingsJSON);
    editor.apply();
  }

  public String getSettings() {
    return sharedPreferences.getString(context.getString(R.string.settings_json_key), "");
  }

  public void saveLogs() {
    SharedPreferences.Editor editor = sharedPreferences.edit();
    Gson gson = new Gson();
    String msgJSON = gson.toJson(AppLogging.getInstance().getLogs());
    editor.putString(context.getString(R.string.logs_json_key), msgJSON);
    editor.apply();
  }

  public void removeExistingLogs() {
    SharedPreferences.Editor editor = sharedPreferences.edit();
    editor.remove(context.getString(R.string.logs_json_key));
    editor.apply();
  }

  public ArrayList<AppLog> getSavedLogs() {
    String logsJSONString =
        sharedPreferences.getString(context.getString(R.string.logs_json_key), null);
    if (logsJSONString == null || logsJSONString.isEmpty()) {
      return new ArrayList<>();
    }
    Gson gson = new Gson();
    Type type = new TypeToken<ArrayList<AppLog>>() {}.getType();
    ArrayList<AppLog> appLogs = gson.fromJson(logsJSONString, type);
    if (appLogs == null) {
      return new ArrayList<>();
    }
    return appLogs;
  }
}
