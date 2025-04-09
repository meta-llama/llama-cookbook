package com.example.llamaandroiddemo

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button

class HomescreenActivity : AppCompatActivity() {

    private lateinit var startChatButton: Button

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