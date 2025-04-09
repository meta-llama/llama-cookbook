/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.llamaandroiddemo;

import android.net.Uri;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.Collections;

import io.noties.markwon.Markwon;

public class MessageAdapter extends ArrayAdapter<Message> {

  private final ArrayList<Message> savedMessages;

  public MessageAdapter(
      android.content.Context context, int resource, ArrayList<Message> savedMessages) {
    super(context, resource);
    this.savedMessages = savedMessages;
  }

  @Override
  public View getView(int position, View convertView, ViewGroup parent) {
    Message currentMessage = getItem(position);
    int layoutIdForListItem;

    if (currentMessage.getMessageType() == MessageType.SYSTEM) {
      layoutIdForListItem = R.layout.system_message;
    } else {
      layoutIdForListItem =
          currentMessage.getIsSent() ? R.layout.sent_message : R.layout.received_message;
    }
    View listItemView =
        LayoutInflater.from(getContext()).inflate(layoutIdForListItem, parent, false);
    if (currentMessage.getMessageType() == MessageType.IMAGE) {
      ImageView messageImageView = listItemView.requireViewById(R.id.message_image);
      messageImageView.setImageURI(Uri.parse(currentMessage.getImagePath()));
      TextView messageTextView = listItemView.requireViewById(R.id.message_text);
      messageTextView.setVisibility(View.GONE);
    } else {
      TextView messageTextView = listItemView.requireViewById(R.id.message_text);
      String markdownString = currentMessage.getText();
      Markwon markwon = Markwon.create(this.getContext());
      markwon.setMarkdown(messageTextView, markdownString);
    }

    String metrics = "";
    TextView tokensView;
    if (currentMessage.getTokensPerSecond() > 0) {
      metrics = String.format("%.2f", currentMessage.getTokensPerSecond()) + "t/s  ";
    }

    if (currentMessage.getTotalGenerationTime() > 0) {
      metrics = metrics + (float) currentMessage.getTotalGenerationTime() / 1000 + "s  ";
    }

    if (currentMessage.getTokensPerSecond() > 0 || currentMessage.getTotalGenerationTime() > 0) {
      tokensView = listItemView.requireViewById(R.id.generation_metrics);
      tokensView.setText(metrics);
      TextView separatorView = listItemView.requireViewById(R.id.bar);
      separatorView.setVisibility(View.VISIBLE);
    }

    if (currentMessage.getTimestamp() > 0) {
      TextView timestampView = listItemView.requireViewById(R.id.timestamp);
      timestampView.setText(currentMessage.getFormattedTimestamp());
    }

    return listItemView;
  }

  @Override
  public void add(Message msg) {
    super.add(msg);
    savedMessages.add(msg);
  }

  @Override
  public void clear() {
    super.clear();
    savedMessages.clear();
  }

  public ArrayList<Message> getSavedMessages() {
    return savedMessages;
  }

  public ArrayList<Message> getRecentSavedTextMessages(int numOfLatestPromptMessages) {
    ArrayList<Message> recentMessages = new ArrayList<Message>();

    // We don't want the "last" empty prompt response message that is yet to generate.
    // -1 to get last index and another -1 to exclude the above. Therefore -2.
    int lastIndex = savedMessages.size() - 2;

    // In most cases lastIndex >=0 .
    // A situation where the user clears chat history and enters prompt. Causes lastIndex=-1 .
    if (lastIndex >= 0) {
      Message messageToAdd = savedMessages.get(lastIndex);
      int oldPromptID = messageToAdd.getPromptID();

      for (int i = 0; i < savedMessages.size() - 1; i++) {
        messageToAdd = savedMessages.get(lastIndex - i);
        if (messageToAdd.getMessageType() != MessageType.SYSTEM) {
          if (messageToAdd.getPromptID() != oldPromptID) {
            numOfLatestPromptMessages--;
            oldPromptID = messageToAdd.getPromptID();
          }
          if (numOfLatestPromptMessages >= 0) {
            if (messageToAdd.getMessageType() == MessageType.TEXT && !messageToAdd.getText().isEmpty()) {
              recentMessages.add(messageToAdd);
            }
            if (messageToAdd.getMessageType() == MessageType.IMAGE && !messageToAdd.getImagePath().isEmpty()) {
              recentMessages.add(messageToAdd);
            }
          } else {
            break;
          }
        }
      }
      // To place the order in [input1, output1, input2, output2...]
      Collections.reverse(recentMessages);
    }

    return recentMessages;
  }

  public int getMaxPromptID() {
    int maxPromptID = -1;
    for (Message msg : savedMessages) {

      maxPromptID = Math.max(msg.getPromptID(), maxPromptID);
    }
    return maxPromptID;
  }
}
