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

public class Message {
  private String text;
  private final boolean isSent;
  private float tokensPerSecond;
  private long totalGenerationTime;
  private final long timestamp;
  private final MessageType messageType;
  private String imagePath;
  private final int promptID;

  private static final String TIMESTAMP_FORMAT = "hh:mm a"; // example: 2:23 PM

  public Message(String text, boolean isSent, MessageType messageType, int promptID) {
    this.isSent = isSent;
    this.messageType = messageType;
    this.promptID = promptID;

    if (messageType == MessageType.IMAGE) {
      this.imagePath = text;
    } else {
      this.text = text;
    }

    if (messageType != MessageType.SYSTEM) {
      this.timestamp = System.currentTimeMillis();
    } else {
      this.timestamp = (long) 0;
    }
  }

  public int getPromptID() {
    return promptID;
  }

  public MessageType getMessageType() {
    return messageType;
  }

  public String getImagePath() {
    return imagePath;
  }

  public String getText() {
    return text;
  }

  public void appendText(String text) {
    this.text += text;
  }

  public void setText(String text) {
    this.text = text;
  }

  public boolean getIsSent() {
    return isSent;
  }

  public void setTokensPerSecond(float tokensPerSecond) {
    this.tokensPerSecond = tokensPerSecond;
  }

  public void setTotalGenerationTime(long totalGenerationTime) {
    this.totalGenerationTime = totalGenerationTime;
  }

  public float getTokensPerSecond() {
    return tokensPerSecond;
  }

  public long getTotalGenerationTime() {
    return totalGenerationTime;
  }

  public long getTimestamp() {
    return timestamp;
  }

  public String getFormattedTimestamp() {
    SimpleDateFormat formatter = new SimpleDateFormat(TIMESTAMP_FORMAT, Locale.getDefault());
    Date date = new Date(timestamp);
    return formatter.format(date);
  }
}
