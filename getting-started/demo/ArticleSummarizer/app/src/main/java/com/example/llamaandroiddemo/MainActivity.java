/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.llamaandroiddemo;

import android.Manifest;
import android.app.ActivityManager;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Process;
import android.provider.MediaStore;
import android.system.ErrnoException;
import android.system.Os;
import android.util.Log;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.PickVisualMediaRequest;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity implements Runnable, InferenceStreamingCallback {
  private EditText mEditTextMessage;
  private ImageButton mSendButton;
  private ImageButton mGalleryButton;
  private ImageButton mCameraButton;
  private ListView mMessagesView;
  private MessageAdapter mMessageAdapter;
  private Message mResultMessage = null;
  private ImageButton mSettingsButton;
  private TextView mMemoryView;
  private ActivityResultLauncher<PickVisualMediaRequest> mPickGallery;
  private ActivityResultLauncher<Uri> mCameraRoll;
  private List<Uri> mSelectedImageUri;
  private ConstraintLayout mMediaPreviewConstraintLayout;
  private LinearLayout mAddMediaLayout;
  private static final int MAX_NUM_OF_IMAGES = 5;
  private static final int REQUEST_IMAGE_CAPTURE = 1;
  private TextView mGenerationModeButton;
  private Uri cameraImageUri;
  private DemoSharedPreferences mDemoSharedPreferences;
  private SettingsFields mCurrentSettingsFields;
  private Handler mMemoryUpdateHandler;
  private Runnable memoryUpdater;
  private int promptID = 0;
  private Executor executor;
  private ExampleLlamaRemoteInference exampleLlamaRemoteInference;
  private LanguageSelector languageSelector;
  private void populateExistingMessages(String existingMsgJSON) {
    Gson gson = new Gson();
    Type type = new TypeToken<ArrayList<Message>>() {}.getType();
    ArrayList<Message> savedMessages = gson.fromJson(existingMsgJSON, type);
    for (Message msg : savedMessages) {
      mMessageAdapter.add(msg);
    }
    mMessageAdapter.notifyDataSetChanged();
  }

  private int setPromptID() {
    return mMessageAdapter.getMaxPromptID() + 1;
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
	  getWindow().setStatusBarColor(ContextCompat.getColor(this, R.color.status_bar));
	  getWindow().setNavigationBarColor(ContextCompat.getColor(this, R.color.nav_bar));

	  try {
      Os.setenv("ADSP_LIBRARY_PATH", getApplicationInfo().nativeLibraryDir, true);
      Os.setenv("LD_LIBRARY_PATH", getApplicationInfo().nativeLibraryDir, true);
    } catch (ErrnoException e) {
      finish();
    }

    mEditTextMessage = requireViewById(R.id.editTextMessage);
    mSendButton = requireViewById(R.id.sendButton);
    mSendButton.setEnabled(true);
    mMessagesView = requireViewById(R.id.messages_view);
    mMessageAdapter = new MessageAdapter(this, R.layout.sent_message, new ArrayList<Message>());
    mMessagesView.setAdapter(mMessageAdapter);
    mDemoSharedPreferences = new DemoSharedPreferences(this.getApplicationContext());
    String existingMsgJSON = mDemoSharedPreferences.getSavedMessages();
    if (!existingMsgJSON.isEmpty()) {
      populateExistingMessages(existingMsgJSON);
      promptID = setPromptID();
    }
    mSettingsButton = requireViewById(R.id.settings);
    mSettingsButton.setOnClickListener(
            view -> {
              Intent myIntent = new Intent(MainActivity.this, SettingsActivity.class);
              MainActivity.this.startActivity(myIntent);
            });

    mCurrentSettingsFields = new SettingsFields();
    mMemoryUpdateHandler = new Handler(Looper.getMainLooper());
    setupMediaButton();
    setupGalleryPicker();
    setupCameraRoll();
    startMemoryUpdate();
    setupShowLogsButton();
    setupLanguageSelector();
    executor = Executors.newSingleThreadExecutor();
    onModelRunStopped();
  }

  private void setupLanguageSelector() {
    languageSelector = new LanguageSelector(this);
    ImageButton button = findViewById(R.id.addLanguageButton);
    button.setOnClickListener(
            view -> {
              Log.d("hello","you click language button");
              languageSelector.showLanguageSelector();
            }
    );
  }

  @Override
  protected void onPause() {
    super.onPause();
    mDemoSharedPreferences.addMessages(mMessageAdapter);
  }

  @Override
  protected void onResume() {
    super.onResume();
    // Check for if settings parameters have changed
    AppLogging.getInstance().log("onResume is called");
    Gson gson = new Gson();
    String settingsFieldsJSON = mDemoSharedPreferences.getSettings();
    if (!settingsFieldsJSON.isEmpty()) {
      SettingsFields updatedSettingsFields =
              gson.fromJson(settingsFieldsJSON, SettingsFields.class);
      if (updatedSettingsFields == null) {
        // Added this check, because gson.fromJson can return null
        askUserToSelectModel();
        return;
      }
      AppLogging.getInstance().log("test "+ updatedSettingsFields.getRemoteURL());

      boolean isUpdated = !mCurrentSettingsFields.equals(updatedSettingsFields);
      boolean isLoadModel = updatedSettingsFields.getIsLoadModel();
      if (isUpdated || isLoadModel) {
        if (!mCurrentSettingsFields.getRemoteURL().equals(updatedSettingsFields.getRemoteURL())) {
          // Remote URL changes
          AppLogging.getInstance().log(mCurrentSettingsFields.getRemoteURL() + "remote URL is changing to " + updatedSettingsFields.getRemoteURL());
          exampleLlamaRemoteInference = new ExampleLlamaRemoteInference(updatedSettingsFields.getRemoteURL());
        }

        AppLogging.getInstance().log("llamaRemoteInference " + (exampleLlamaRemoteInference == null));

        if (exampleLlamaRemoteInference == null) {
          askUserToSelectModel();
        } else {
          String message = "Remote inference with model: " + updatedSettingsFields.getRemoteModel();
          addSystemMessage(message);
        }

        checkForClearChatHistory(updatedSettingsFields);
        // Update current to point to the latest
        mCurrentSettingsFields = new SettingsFields(updatedSettingsFields);
        AppLogging.getInstance().log("onResume mCurrentSettingsFields " + mCurrentSettingsFields);
      }
    } else {
      askUserToSelectModel();
    }
  }

  private void checkForClearChatHistory(SettingsFields updatedSettingsFields) {
    if (updatedSettingsFields.getIsClearChatHistory()) {
      mMessageAdapter.clear();
      mMessageAdapter.notifyDataSetChanged();
      mDemoSharedPreferences.removeExistingMessages();
      // changing to false since chat history has been cleared.
      updatedSettingsFields.saveIsClearChatHistory(false);
      mDemoSharedPreferences.addSettings(updatedSettingsFields);
    }
  }

  private void addSystemMessage(String message) {
    Message systemMessage = new Message(message, false, MessageType.SYSTEM, 0);
    AppLogging.getInstance().log(message);
    runOnUiThread(
            () -> {
              mMessageAdapter.add(systemMessage);
              mMessageAdapter.notifyDataSetChanged();
            });
  }

  private void askUserToSelectModel() {
    String askLoadModel =
            "To get started, configure remote URL" +
            "from the top right corner";
    addSystemMessage(askLoadModel);
  }

  private void setupShowLogsButton() {
    ImageButton showLogsButton = requireViewById(R.id.showLogsButton);
    showLogsButton.setOnClickListener(
            view -> {
              Intent myIntent = new Intent(MainActivity.this, LogsActivity.class);
              MainActivity.this.startActivity(myIntent);
            });
  }

  private void setupMediaButton() {
    mAddMediaLayout = requireViewById(R.id.addMediaLayout);
    mAddMediaLayout.setVisibility(View.GONE); // We hide this initially

    ImageButton addMediaButton = requireViewById(R.id.addMediaButton);
    addMediaButton.setOnClickListener(
            view -> {
              mAddMediaLayout.setVisibility(View.VISIBLE);
            });

    mGalleryButton = requireViewById(R.id.galleryButton);
    mGalleryButton.setOnClickListener(
            view -> {
              // Launch the photo picker and let the user choose only images.
              mPickGallery.launch(
                      new PickVisualMediaRequest.Builder()
                              .setMediaType(ActivityResultContracts.PickVisualMedia.ImageOnly.INSTANCE)
                              .build());
            });
    mCameraButton = requireViewById(R.id.cameraButton);
    mCameraButton.setOnClickListener(
            view -> {
              Log.d("CameraRoll", "Check permission");
              if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA)
                          != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(
                        MainActivity.this,
                        new String[] {Manifest.permission.CAMERA},
                        REQUEST_IMAGE_CAPTURE);
              } else {
                launchCamera();
              }
            });
  }

  private void setupCameraRoll() {
    // Registers a camera roll activity launcher.
    mCameraRoll =
            registerForActivityResult(
                    new ActivityResultContracts.TakePicture(),
                    result -> {
                      if (result && cameraImageUri != null) {
                        Log.d("CameraRoll", "Photo saved to uri: " + cameraImageUri);
                        mAddMediaLayout.setVisibility(View.GONE);
                        List<Uri> uris = new ArrayList<>();
                        uris.add(cameraImageUri);
                        showMediaPreview(uris);
                      } else {
                        // Delete the temp image file based on the url since the photo is not successfully taken
                        if (cameraImageUri != null) {
                          ContentResolver contentResolver = MainActivity.this.getContentResolver();
                          contentResolver.delete(cameraImageUri, null, null);
                          Log.d("CameraRoll", "No photo taken. Delete temp uri");
                        }
                      }
                    });
    mMediaPreviewConstraintLayout = requireViewById(R.id.mediaPreviewConstraintLayout);
    ImageButton mediaPreviewCloseButton = requireViewById(R.id.mediaPreviewCloseButton);
    mediaPreviewCloseButton.setOnClickListener(
            view -> {
              mMediaPreviewConstraintLayout.setVisibility(View.GONE);
              mSelectedImageUri = null;
            });

    ImageButton addMoreImageButton = requireViewById(R.id.addMoreImageButton);
    addMoreImageButton.setOnClickListener(
            view -> {
              Log.d("addMore", "clicked");
              mMediaPreviewConstraintLayout.setVisibility(View.GONE);
              // Direct user to select type of input
              mCameraButton.callOnClick();
            });
  }

  private String updateMemoryUsage() {
    ActivityManager.MemoryInfo memoryInfo = new ActivityManager.MemoryInfo();
    ActivityManager activityManager = (ActivityManager) getSystemService(ACTIVITY_SERVICE);
    if (activityManager == null) {
      return "---";
    }
    activityManager.getMemoryInfo(memoryInfo);
    long totalMem = memoryInfo.totalMem / (1024 * 1024);
    long availableMem = memoryInfo.availMem / (1024 * 1024);
    long usedMem = totalMem - availableMem;
    return usedMem + "MB";
  }

  private void startMemoryUpdate() {
    mMemoryView = requireViewById(R.id.ram_usage_live);
    memoryUpdater =
            new Runnable() {
              @Override
              public void run() {
                mMemoryView.setText(updateMemoryUsage());
                mMemoryUpdateHandler.postDelayed(this, 1000);
              }
            };
    mMemoryUpdateHandler.post(memoryUpdater);
  }

  @Override
  public void onRequestPermissionsResult(
          int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    if (requestCode == REQUEST_IMAGE_CAPTURE && grantResults.length != 0) {
      if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
        launchCamera();
      } else if (grantResults[0] == PackageManager.PERMISSION_DENIED) {
        Log.d("CameraRoll", "Permission denied");
      }
    }
  }

  private void launchCamera() {
    ContentValues values = new ContentValues();
    values.put(MediaStore.Images.Media.TITLE, "New Picture");
    values.put(MediaStore.Images.Media.DESCRIPTION, "From Camera");
    values.put(MediaStore.Images.Media.RELATIVE_PATH, "DCIM/Camera/");
    cameraImageUri =
            MainActivity.this
                    .getContentResolver()
                    .insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
    mCameraRoll.launch(cameraImageUri);
  }

  private void setupGalleryPicker() {
    // Registers a photo picker activity launcher in single-select mode.
    mPickGallery =
            registerForActivityResult(
                    new ActivityResultContracts.PickMultipleVisualMedia(MAX_NUM_OF_IMAGES),
                    uris -> {
                      if (!uris.isEmpty()) {
                        Log.d("PhotoPicker", "Selected URIs: " + uris);
                        mAddMediaLayout.setVisibility(View.GONE);
                        for (Uri uri : uris) {
                          MainActivity.this
                                  .getContentResolver()
                                  .takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION);
                        }
                        showMediaPreview(uris);
                      } else {
                        Log.d("PhotoPicker", "No media selected");
                      }
                    });

    mMediaPreviewConstraintLayout = requireViewById(R.id.mediaPreviewConstraintLayout);
    ImageButton mediaPreviewCloseButton = requireViewById(R.id.mediaPreviewCloseButton);
    mediaPreviewCloseButton.setOnClickListener(
            view -> {
              mMediaPreviewConstraintLayout.setVisibility(View.GONE);
              mSelectedImageUri = null;
            });

    ImageButton addMoreImageButton = requireViewById(R.id.addMoreImageButton);
    addMoreImageButton.setOnClickListener(
            view -> {
              Log.d("addMore", "clicked");
              mMediaPreviewConstraintLayout.setVisibility(View.GONE);
              mGalleryButton.callOnClick();
            });
  }

  private void showMediaPreview(List<Uri> uris) {
    if (mSelectedImageUri == null) {
      mSelectedImageUri = uris;
    } else {
      mSelectedImageUri.addAll(uris);
    }

    if (mSelectedImageUri.size() > MAX_NUM_OF_IMAGES) {
      mSelectedImageUri = mSelectedImageUri.subList(0, MAX_NUM_OF_IMAGES);
      Toast.makeText(
                      this, "Only max " + MAX_NUM_OF_IMAGES + " images are allowed", Toast.LENGTH_SHORT)
              .show();
    }
    Log.d("mSelectedImageUri", mSelectedImageUri.size() + " " + mSelectedImageUri);

    mMediaPreviewConstraintLayout.setVisibility(View.VISIBLE);

    List<ImageView> imageViews = new ArrayList<ImageView>();

    // Pre-populate all the image views that are available from the layout (currently max 5)
    imageViews.add(requireViewById(R.id.mediaPreviewImageView1));
    imageViews.add(requireViewById(R.id.mediaPreviewImageView2));
    imageViews.add(requireViewById(R.id.mediaPreviewImageView3));
    imageViews.add(requireViewById(R.id.mediaPreviewImageView4));
    imageViews.add(requireViewById(R.id.mediaPreviewImageView5));

    // Hide all the image views (reset state)
    for (int i = 0; i < imageViews.size(); i++) {
      imageViews.get(i).setVisibility(View.GONE);
    }

    // Only show/render those that have proper Image URIs
    for (int i = 0; i < mSelectedImageUri.size(); i++) {
      imageViews.get(i).setVisibility(View.VISIBLE);
      imageViews.get(i).setImageURI(mSelectedImageUri.get(i));
    }
  }

  private void addSelectedImagesToChatThread(List<Uri> selectedImageUri) {
    if (selectedImageUri == null) {
      return;
    }
    mMediaPreviewConstraintLayout.setVisibility(View.GONE);
    for (int i = 0; i < selectedImageUri.size(); i++) {
      Uri imageURI = selectedImageUri.get(i);
      Log.d("image uri ", "test " + imageURI.getPath());
      mMessageAdapter.add(new Message(imageURI.toString(), true, MessageType.IMAGE, 0));
    }
    mMessageAdapter.notifyDataSetChanged();
  }

  private void onModelRunStopped() {
    mSendButton.setClickable(true);
    mSendButton.setImageResource(R.drawable.baseline_send_24);
    mSendButton.setOnClickListener(
            view -> {
              try {
                InputMethodManager imm = (InputMethodManager) getSystemService(INPUT_METHOD_SERVICE);
                imm.hideSoftInputFromWindow(Objects.requireNonNull(getCurrentFocus()).getWindowToken(), 0);
              } catch (Exception e) {
                AppLogging.getInstance().log("Keyboard dismissal error: " + e.getMessage());
              }
              addSelectedImagesToChatThread(mSelectedImageUri);
              String rawPrompt = mEditTextMessage.getText().toString();
              mMessageAdapter.add(new Message(rawPrompt, true, MessageType.TEXT, promptID));
              mMessageAdapter.notifyDataSetChanged();
              mEditTextMessage.setText("");
              mResultMessage = new Message("", false, MessageType.TEXT, promptID);
              mMessageAdapter.add(mResultMessage);
              // Scroll to bottom of the list
              mMessagesView.smoothScrollToPosition(mMessageAdapter.getCount() - 1);
              // After images are added to prompt and chat thread, we clear the imageURI list
              // Note: This has to be done after imageURIs are no longer needed by LlamaModule
              mSelectedImageUri = null;
              promptID++;
              Runnable runnable =
                      new Runnable() {
                        @Override
                        public void run() {
                          Process.setThreadPriority(Process.THREAD_PRIORITY_MORE_FAVORABLE);
                          long generateStartTime = System.currentTimeMillis();
                          remoteLlamaGeneration(rawPrompt);

                          long generateDuration = System.currentTimeMillis() - generateStartTime;
                          mResultMessage.setTotalGenerationTime(generateDuration);
                          runOnUiThread(
                                  new Runnable() {
                                    @Override
                                    public void run() {
                                      onModelRunStopped();
                                    }
                                  });
                          AppLogging.getInstance().log("Inference completed");
                        }
                      };
              executor.execute(runnable);
            });
    mMessageAdapter.notifyDataSetChanged();
  }

  public void remoteLlamaGeneration(String rawPrompt) {
    AppLogging.getInstance().log("Running inference remotely ("+ mCurrentSettingsFields.getRemoteModel() +").. raw prompt=" + rawPrompt);
    String systemPrompt = mCurrentSettingsFields.getSystemPrompt();
    String modelName = mCurrentSettingsFields.getRemoteModel();
    double temperature = mCurrentSettingsFields.getTemperature();

    String result = "";

    // Get user selected language, and add that to system prompt
    systemPrompt += " Summarize and translate it into the following language: " + languageSelector.getSelectedLanguage();
    Log.d("updated system prompt with language selected",systemPrompt);

    result = exampleLlamaRemoteInference.inferenceStartWithoutAgent(
              modelName,
              temperature,
              mMessageAdapter.getRecentSavedTextMessages(AppUtils.CONVERSATION_HISTORY_MESSAGE_LOOKBACK),
              systemPrompt,
              this
    );
    mResultMessage.appendText(result);
  }

  @Override
  public void run() {
    runOnUiThread(
            new Runnable() {
              @Override
              public void run() {
                mMessageAdapter.notifyDataSetChanged();
              }
            });
  }

  @Override
  public void onBackPressed() {
    super.onBackPressed();
    if (mAddMediaLayout != null && mAddMediaLayout.getVisibility() == View.VISIBLE) {
      mAddMediaLayout.setVisibility(View.GONE);
    } else {
      // Default behavior of back button
      finish();
    }
  }

  @Override
  protected void onDestroy() {
    super.onDestroy();
    mMemoryUpdateHandler.removeCallbacks(memoryUpdater);
    // This is to cover the case where the app is shutdown when user is on MainActivity but
    // never clicked on the logsActivity
    AppLogging.getInstance().saveLogs();
  }

  @Override
  public void onStreamReceived(@NonNull String message) {
    AppLogging.getInstance().log("this is stream received: " + message);
    runOnUiThread(
            () -> {
              mResultMessage.appendText(message);
              mMessageAdapter.notifyDataSetChanged();
            });
  }

  @Override
  public void onStatStreamReceived(float tps) {
    AppLogging.getInstance().log("this is stats received: " + tps);
    runOnUiThread(
            () -> {
              mResultMessage.setTokensPerSecond(tps);
              mMessageAdapter.notifyDataSetChanged();
            });
  }
}
