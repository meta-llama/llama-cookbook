package com.example.llamaandroiddemo

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.TextView

class HomescreenActivity : AppCompatActivity() {

    private lateinit var startChatButton: Button
    private lateinit var introTextView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_homescreen)

        // Initialize UI components
        startChatButton = findViewById(R.id.btn_start_chat)

        // Set up start chat button click listener
        startChatButton.setOnClickListener {
            val intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
        }
    }
}