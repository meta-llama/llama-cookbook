/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

plugins {
  id("com.android.application")
  id("org.jetbrains.kotlin.android")
}

android {
  namespace = "com.example.llamaandroiddemo"
  compileSdk = 34

  defaultConfig {
    applicationId = "com.example.llamaandroiddemo"
    minSdk = 28
    targetSdk = 33
    versionCode = 1
    versionName = "1.0"

    testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    vectorDrawables { useSupportLibrary = true }
    externalNativeBuild { cmake { cppFlags += "" } }
    packaging {
      resources.excludes.add("META-INF/DEPENDENCIES")
    }
  }

  buildTypes {
    release {
      isMinifyEnabled = false
      proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
    }
  }
  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_1_8
    targetCompatibility = JavaVersion.VERSION_1_8
  }
  kotlinOptions { jvmTarget = "1.8" }
  buildFeatures { compose = true }
  composeOptions { kotlinCompilerExtensionVersion = "1.4.3" }
  packaging { resources { excludes += "/META-INF/{AL2.0,LGPL2.1}" } }
}

dependencies {
  implementation("androidx.core:core-ktx:1.9.0")
  implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.6.1")
  implementation("androidx.activity:activity-compose:1.7.0")
  implementation(platform("androidx.compose:compose-bom:2023.03.00"))
  implementation("androidx.compose.ui:ui")
  implementation("androidx.compose.ui:ui-graphics")
  implementation("androidx.compose.ui:ui-tooling-preview")
  implementation("androidx.compose.material3:material3")
  implementation("androidx.appcompat:appcompat:1.6.1")
  implementation("androidx.camera:camera-core:1.3.0-rc02")
  implementation("androidx.constraintlayout:constraintlayout:2.2.0-alpha12")
  implementation("com.facebook.fbjni:fbjni:0.5.1")
  implementation("com.google.code.gson:gson:2.8.6")
  implementation("com.google.android.material:material:1.12.0")
  implementation("androidx.activity:activity:1.9.0")
  testImplementation("junit:junit:4.13.2")
  androidTestImplementation("androidx.test.ext:junit:1.1.5")
  androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
  androidTestImplementation(platform("androidx.compose:compose-bom:2023.03.00"))
  androidTestImplementation("androidx.compose.ui:ui-test-junit4")
  debugImplementation("androidx.compose.ui:ui-tooling")
  debugImplementation("androidx.compose.ui:ui-test-manifest")
  implementation("com.squareup.okhttp3:okhttp:4.10.0")
  implementation("com.google.guava:guava:31.0-jre")
  implementation("com.fasterxml.jackson.module:jackson-module-kotlin:2.18.1")
  implementation("com.fasterxml.jackson.datatype:jackson-datatype-jdk8:2.18.1")
  implementation("com.fasterxml.jackson.datatype:jackson-datatype-jsr310:2.18.1")
  implementation("org.jetbrains.kotlinx:kotlinx-datetime:0.2.1")
  implementation("io.noties.markwon:core:4.6.2")
}