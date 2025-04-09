/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.llamaandroiddemo;

import android.app.Application;
import android.util.Log;
import java.util.ArrayList;

public class AppLogging extends Application {
  private static AppLogging singleton;

  private ArrayList<AppLog> logs;
  private DemoSharedPreferences mDemoSharedPreferences;

  @Override
  public void onCreate() {
    super.onCreate();
    singleton = this;
    mDemoSharedPreferences = new DemoSharedPreferences(this.getApplicationContext());
    logs = mDemoSharedPreferences.getSavedLogs();
    if (logs == null) { // We don't have existing sharedPreference stored
      logs = new ArrayList<>();
    }
  }

  public static AppLogging getInstance() {
    return singleton;
  }

  public void log(String message) {
    AppLog appLog = new AppLog(message);
    logs.add(appLog);
    Log.d("AppLogging", appLog.getMessage());
  }

  public ArrayList<AppLog> getLogs() {
    return logs;
  }

  public void clearLogs() {
    logs.clear();
    mDemoSharedPreferences.removeExistingLogs();
  }

  public void saveLogs() {
    mDemoSharedPreferences.saveLogs();
  }
}
