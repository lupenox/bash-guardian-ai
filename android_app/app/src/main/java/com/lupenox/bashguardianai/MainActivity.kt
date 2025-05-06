package com.lupenox.bashguardianai
import android.speech.tts.TextToSpeech
import java.util.Locale
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.lupenox.bashguardianai.ui.BashCompanionUI
import com.lupenox.bashguardianai.ui.theme.BashGuardianAITheme

class MainActivity : ComponentActivity() {
    private lateinit var tts: TextToSpeech

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            tts = TextToSpeech(this) { status ->
                if (status == TextToSpeech.SUCCESS) {
                    tts.language = Locale.US
                    tts.setSpeechRate(0.9f) // slower for cozy vibes
                }
            }

            BashGuardianAITheme {
                BashCompanionUI()
            }
        }
    }
}
