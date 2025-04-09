/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.llamaandroiddemo;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class AppLog {
  private final Long timestamp;
  private final String message;

  public AppLog(String message) {
    this.timestamp = getCurrentTimeStamp();
    this.message = message;
  }

  public Long getTimestamp() {
    return timestamp;
  }

  public String getMessage() {
    return message;
  }

  public String getFormattedLog() {
    return "[" + getFormattedTimeStamp() + "] " + message;
  }

  private Long getCurrentTimeStamp() {
    return System.currentTimeMillis();
  }

  private String getFormattedTimeStamp() {
    return formatDate(timestamp);
  }

  private String formatDate(long milliseconds) {
    SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd  HH:mm:ss", Locale.getDefault());
    Date date = new Date(milliseconds);
    return formatter.format(date);
  }
}
