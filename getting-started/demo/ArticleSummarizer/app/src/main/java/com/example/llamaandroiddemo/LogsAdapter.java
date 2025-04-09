/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.llamaandroiddemo;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.TextView;
import androidx.annotation.NonNull;
import java.util.Objects;

public class LogsAdapter extends ArrayAdapter<AppLog> {
  public LogsAdapter(android.content.Context context, int resource) {
    super(context, resource);
  }

  static class ViewHolder {
    private TextView logTextView;
  }

  @NonNull
  @Override
  public View getView(int position, View convertView, @NonNull ViewGroup parent) {
    ViewHolder mViewHolder = null;

    String logMessage = Objects.requireNonNull(getItem(position)).getFormattedLog();

    if (convertView == null || convertView.getTag() == null) {
      mViewHolder = new ViewHolder();
      convertView = LayoutInflater.from(getContext()).inflate(R.layout.logs_message, parent, false);
      mViewHolder.logTextView = convertView.requireViewById(R.id.logsTextView);
    } else {
      mViewHolder = (ViewHolder) convertView.getTag();
    }
    mViewHolder.logTextView.setText(logMessage);
    return convertView;
  }
}
