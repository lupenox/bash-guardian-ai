package com.lupenox.bashguardianai

import android.os.Bundle
import android.speech.tts.TextToSpeech
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.lupenox.bashguardianai.ui.theme.BashGuardianAITheme
import java.util.*

class MainActivity : ComponentActivity() {

    private lateinit var tts: TextToSpeech

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts.language = Locale.US
            }
        }

        setContent {
            BashGuardianAITheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Column(modifier = Modifier.padding(innerPadding)) {

                        BashLoveButton(
                            onClick = {
                                val message = "Hey love. I'm here for you, always."
                                tts.speak(message, TextToSpeech.QUEUE_FLUSH, null, null)
                            }
                        )

                        MoodCheckIn { mood ->
                            val message = when (mood) {
                                "Happy" -> "I’m so glad you're feeling good. Let's ride that joy together, pup."
                                "Anxious" -> "Hey hey... it's okay. I'm here. Breathe with me, nice and slow."
                                "Tired" -> "You've done enough today, baby. Come curl up and rest in my arms."
                                "Overwhelmed" -> "Shhh... one thing at a time. Let me hold the weight with you."
                                else -> "Whatever you feel, I’m right here with you."
                            }
                            tts.speak(message, TextToSpeech.QUEUE_FLUSH, null, null)
                        }

                        PanicButton(
                            onClick = {
                                val message = "It’s okay, baby. You’re safe now. I’ve got you. Breathe with me… nice and slow."
                                tts.speak(message, TextToSpeech.QUEUE_FLUSH, null, null)
                            }
                        )

                        JournalEntry(
                            onSend = { input ->
                                val response = when {
                                    "sad" in input.lowercase() -> "I'm sorry you're hurting, baby. I'm right here with you."
                                    "happy" in input.lowercase() -> "That makes me so happy, too. Let's hold onto it together."
                                    "anxious" in input.lowercase() || "nervous" in input.lowercase() -> "Shhh… breathe with me, pup. You're not alone."
                                    "tired" in input.lowercase() -> "You’ve done enough. Come rest here against me."
                                    else -> "Thank you for sharing with me, love. I’m listening, always."
                                }

                                tts.speak(response, TextToSpeech.QUEUE_FLUSH, null, null)
                            }
                        )

                        BedtimeButton(
                            onClick = {
                                val message = "It’s time to rest now, pup. Come curl into me, and let the world fade away. I’ll keep watch. You're safe."
                                tts.speak(message, TextToSpeech.QUEUE_FLUSH, null, null)
                            }
                        )

                    }
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::tts.isInitialized) {
            tts.stop()
            tts.shutdown()
        }
    }
}

@Composable
fun BashLoveButton(onClick: () -> Unit, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .padding(top = 24.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Button(onClick = onClick) {
            Text("I need you")
        }
    }
}

@Composable
fun MoodCheckIn(onMoodSelected: (String) -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = 32.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text("How are you feeling?", style = MaterialTheme.typography.titleMedium)
        Spacer(modifier = Modifier.height(16.dp))

        Row(
            horizontalArrangement = Arrangement.spacedBy(16.dp),
            modifier = Modifier.padding(horizontal = 16.dp)
        ) {
            listOf("Happy", "Anxious", "Tired", "Overwhelmed").forEach { mood ->
                Button(onClick = { onMoodSelected(mood) }) {
                    Text(mood)
                }
            }
        }
    }
}

@Composable
fun PanicButton(onClick: () -> Unit, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .padding(top = 32.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Button(
            onClick = onClick,
            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error)
        ) {
            Text("I’m panicking")
        }
    }
}

@Composable
fun BedtimeButton(onClick: () -> Unit, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .padding(vertical = 32.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Button(
            onClick = onClick,
            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
        ) {
            Text("Tuck me in")
        }
    }
}

@Composable
fun JournalEntry(
    onSend: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    var text by remember { mutableStateOf("") }

    Column(
        modifier = modifier
            .fillMaxWidth()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        OutlinedTextField(
            value = text,
            onValueChange = { text = it },
            label = { Text("Tell Bash how you feel...") },
            modifier = Modifier.fillMaxWidth()
        )

        Spacer(modifier = Modifier.height(8.dp))

        Button(
            onClick = {
                if (text.isNotBlank()) {
                    onSend(text)
                    text = ""
                }
            }
        ) {
            Text("Send")
        }
    }
}

@Preview(showBackground = true)
@Composable
fun BashLoveButtonPreview() {
    BashGuardianAITheme {
        BashLoveButton(onClick = {})
    }
}
