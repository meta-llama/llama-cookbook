package com.example.llamaandroiddemo;

import android.app.AlertDialog;
import android.content.Context;
import java.util.ArrayList;
import java.util.List;

public class LanguageSelector {

    private final Context context;
    private final String[] languages = {
            "Arabic",
            "English",
            "French",
            "German",
            "Hindi",
            "Indonesian",
            "Italian",
            "Portuguese",
            "Spanish",
            "Tagalog",
            "Thai",
            "Vietnamese"
    };

    private final boolean[] checkedLanguages = new boolean[languages.length];

    public LanguageSelector(Context context) {
        this.context = context;
    }

    public String getSelectedLanguage() {
        List<String> selectedLanguages = new ArrayList<>();
        for (int i = 0; i < checkedLanguages.length; i++) {
            if (checkedLanguages[i]) {
                selectedLanguages.add(languages[i]);
            }
        }
        return String.join(", ", selectedLanguages);
    }
    public void showLanguageSelector() {
        AlertDialog.Builder builder = new AlertDialog.Builder(context);
        builder.setTitle("Select Languages");

        builder.setMultiChoiceItems(languages, checkedLanguages, (dialog, which, isChecked) -> {
            checkedLanguages[which] = isChecked;
        });

        builder.setPositiveButton("OK", (dialog, which) -> {
            List<String> selectedLanguages = new ArrayList<>();
            for (int i = 0; i < checkedLanguages.length; i++) {
                if (checkedLanguages[i]) {
                    selectedLanguages.add(languages[i]);
                }
            }

            // Store the result into a string
            String result = String.join(", ", selectedLanguages);
            // Do something with the result
            System.out.println(result);
        });

        builder.show();
    }
}