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
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import com.google.gson.Gson;
import java.io.File;

public class SettingsActivity extends AppCompatActivity {

  private EditText mSystemPromptEditText;
  private double mSetTemperature;
  private String mSystemPrompt;
  private EditText mRemoteURLEditText;
  private String mRemoteURL;
  public SettingsFields mSettingsFields;
  private String mRemoteModel;
  private TextView mRemoteModelTextView;

  private DemoSharedPreferences mDemoSharedPreferences;
  public static double TEMPERATURE_MIN_VALUE = 0.0;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_settings);
	  getWindow().setStatusBarColor(ContextCompat.getColor(this, R.color.status_bar));
	  getWindow().setNavigationBarColor(ContextCompat.getColor(this, R.color.nav_bar));
	  ViewCompat.setOnApplyWindowInsetsListener(
        requireViewById(R.id.main),
        (v, insets) -> {
          Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
          v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
          return insets;
        });
    mDemoSharedPreferences = new DemoSharedPreferences(getBaseContext());
    mSettingsFields = new SettingsFields();
    setupSettings();
  }

  private void setupSettings() {
    mSystemPromptEditText = requireViewById(R.id.systemPromptText);
    mRemoteURLEditText = requireViewById(R.id.remoteURLEditText);

    loadSettings();

    setupParameterSettings();
    setupPromptSettings();
    setupClearChatHistoryButton();
    setupRemoteInferenceSettings();
  }
  private void setupClearChatHistoryButton() {
    Button clearChatButton = requireViewById(R.id.clearChatButton);
    clearChatButton.setOnClickListener(
        view -> {
          new AlertDialog.Builder(this)
              .setTitle("Delete Chat History")
              .setMessage("Do you really want to delete chat history?")
              .setIcon(android.R.drawable.ic_dialog_alert)
              .setPositiveButton(
                  android.R.string.yes,
                  new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int whichButton) {
                      mSettingsFields.saveIsClearChatHistory(true);
                    }
                  })
              .setNegativeButton(android.R.string.no, null)
              .show();
        });
  }

  private void setupParameterSettings() {
    setupTemperatureSettings();
  }

  private void setupTemperatureSettings() {
    mSetTemperature = mSettingsFields.getTemperature();
    EditText temperatureEditText = requireViewById(R.id.temperatureEditText);
    temperatureEditText.setText(String.valueOf(mSetTemperature));
    temperatureEditText.addTextChangedListener(
        new TextWatcher() {
          @Override
          public void beforeTextChanged(CharSequence s, int start, int count, int after) {}

          @Override
          public void onTextChanged(CharSequence s, int start, int before, int count) {}

          @Override
          public void afterTextChanged(Editable s) {
            mSetTemperature = Double.parseDouble(s.toString());
            // This is needed because temperature is changed together with model loading
            // Once temperature is no longer in LlamaModule constructor, we can remove this
            mSettingsFields.saveLoadModelAction(true);
            saveSettings();
          }
        });
  }

  private void setupRemoteInferenceSettings() {
    mRemoteURL = mSettingsFields.getRemoteURL();
    AppLogging.getInstance().log("mRemoteURL from settings " + mRemoteURL);
    if (mRemoteURL != null) {
      mRemoteURLEditText.setText(mRemoteURL);
    }
    mRemoteURLEditText.addTextChangedListener(
            new TextWatcher() {
              @Override
              public void beforeTextChanged(CharSequence s, int start, int count, int after) {
              }

              @Override
              public void onTextChanged(CharSequence s, int start, int before, int count) {
              }

              @Override
              public void afterTextChanged(Editable s) {
                mRemoteURL = s.toString();
                AppLogging.getInstance().log("after text change remote url" + mRemoteURL);
                mSettingsFields.saveRemoteURL(mRemoteURL);
                saveSettings();
              }
            }
    );

    mRemoteModelTextView = requireViewById(R.id.remoteModelTextView);
    ImageButton mRemoteModelImageButton = requireViewById(R.id.remoteModelImageButton);

    mRemoteModel = mSettingsFields.getRemoteModel();
    AppLogging.getInstance().log("mRemoteModel from settings " + mRemoteModel);
    if (mRemoteModel != null) {
      mRemoteModelTextView.setText(mRemoteModel);
    }

    mRemoteModelImageButton.setOnClickListener(
            view -> {
              String[] models = ModelUtils.getSupportedRemoteModels().toArray(new String[0]);
              AlertDialog.Builder modelBuilder = new AlertDialog.Builder(this);
              modelBuilder.setTitle("Select remote model");
              modelBuilder.setSingleChoiceItems(
                      models,
                      -1,
                      (dialog, item) -> {
                        mRemoteModelTextView.setText(models[item]);
                        mRemoteModel = models[item];
                        dialog.dismiss();
                      });
              modelBuilder.create().show();
            });
  }

  private void setupPromptSettings() {
    setupSystemPromptSettings();
  }

  private void setupSystemPromptSettings() {
    mSystemPrompt = mSettingsFields.getSystemPrompt();
    mSystemPromptEditText.setText(mSystemPrompt);
    mSystemPromptEditText.addTextChangedListener(
        new TextWatcher() {
          @Override
          public void beforeTextChanged(CharSequence s, int start, int count, int after) {}

          @Override
          public void onTextChanged(CharSequence s, int start, int before, int count) {}

          @Override
          public void afterTextChanged(Editable s) {
            mSystemPrompt = s.toString();
          }
        });

    ImageButton resetSystemPrompt = requireViewById(R.id.resetSystemPrompt);
    resetSystemPrompt.setOnClickListener(
        view -> {
          new AlertDialog.Builder(this)
              .setTitle("Reset System Prompt")
              .setMessage("Do you really want to reset system prompt?")
              .setIcon(android.R.drawable.ic_dialog_alert)
              .setPositiveButton(
                  android.R.string.yes,
                  new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int whichButton) {
                      // Clear the messageAdapter and sharedPreference
                      mSystemPromptEditText.setText(PromptFormat.DEFAULT_SYSTEM_PROMPT);
                    }
                  })
              .setNegativeButton(android.R.string.no, null)
              .show();
        });
  }
  private static String[] listLocalFile(String path, String suffix) {
    File directory = new File(path);
    if (directory.exists() && directory.isDirectory()) {
      File[] files = directory.listFiles((dir, name) -> name.toLowerCase().endsWith(suffix));
      String[] result = new String[files.length];
      for (int i = 0; i < files.length; i++) {
        if (files[i].isFile() && files[i].getName().endsWith(suffix)) {
          result[i] = files[i].getAbsolutePath();
        }
      }
      return result;
    }
    return new String[] {};
  }
  private void loadSettings() {
    Gson gson = new Gson();
    String settingsFieldsJSON = mDemoSharedPreferences.getSettings();
    if (!settingsFieldsJSON.isEmpty()) {
      AppLogging.getInstance().log("mSettingsFields " + settingsFieldsJSON);
      mSettingsFields = gson.fromJson(settingsFieldsJSON, SettingsFields.class);
    }
  }

  private void saveSettings() {
    mSettingsFields.saveParameters(mSetTemperature);
    mSettingsFields.savePrompts(mSystemPrompt);
    mSettingsFields.saveRemoteURL(mRemoteURL);
    mSettingsFields.saveRemoteModel(mRemoteModel);
    mDemoSharedPreferences.addSettings(mSettingsFields);
  }

  @Override
  public void onBackPressed() {
    super.onBackPressed();
    saveSettings();
  }
}
