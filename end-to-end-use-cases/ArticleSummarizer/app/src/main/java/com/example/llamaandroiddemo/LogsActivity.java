/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.llamaandroiddemo;

import android.app.AlertDialog;
import android.content.DialogInterface;
import android.os.Build;
import android.os.Bundle;
import android.widget.ImageButton;
import android.widget.ListView;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class LogsActivity extends AppCompatActivity {

  private LogsAdapter mLogsAdapter;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_logs);
    if (Build.VERSION.SDK_INT >= 21) {
      getWindow().setStatusBarColor(ContextCompat.getColor(this, R.color.status_bar));
      getWindow().setNavigationBarColor(ContextCompat.getColor(this, R.color.nav_bar));
    }
    ViewCompat.setOnApplyWindowInsetsListener(
        requireViewById(R.id.main),
        (v, insets) -> {
          Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
          v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
          return insets;
        });

    setupLogs();
    setupClearLogsButton();
  }

  @Override
  public void onResume() {
    super.onResume();
    mLogsAdapter.clear();
    mLogsAdapter.addAll(AppLogging.getInstance().getLogs());
    mLogsAdapter.notifyDataSetChanged();
  }

  private void setupLogs() {
    ListView mLogsListView = requireViewById(R.id.logsListView);
    mLogsAdapter = new LogsAdapter(this, R.layout.logs_message);

    mLogsListView.setAdapter(mLogsAdapter);
    mLogsAdapter.addAll(AppLogging.getInstance().getLogs());
    mLogsAdapter.notifyDataSetChanged();
  }

  private void setupClearLogsButton() {
    ImageButton clearLogsButton = requireViewById(R.id.clearLogsButton);
    clearLogsButton.setOnClickListener(
        view -> {
          new AlertDialog.Builder(this)
              .setTitle("Delete Logs History")
              .setMessage("Do you really want to delete logs history?")
              .setIcon(android.R.drawable.ic_dialog_alert)
              .setPositiveButton(
                  android.R.string.yes,
                  new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int whichButton) {
                      // Clear the messageAdapter and sharedPreference
                      AppLogging.getInstance().clearLogs();
                      mLogsAdapter.clear();
                      mLogsAdapter.notifyDataSetChanged();
                    }
                  })
              .setNegativeButton(android.R.string.no, null)
              .show();
        });
  }

  @Override
  protected void onDestroy() {
    super.onDestroy();
    AppLogging.getInstance().saveLogs();
  }
}
