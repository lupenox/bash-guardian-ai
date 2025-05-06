package com.lupenox.bashguardianai.ui

import android.speech.tts.TextToSpeech
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.lupenox.bashguardianai.api.RetrofitClient
import com.lupenox.bashguardianai.data.BashRequest
import kotlinx.coroutines.launch
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BashCompanionUI() {
    var userInput by remember { mutableStateOf("") }
    val context = LocalContext.current
    val drawerState = rememberDrawerState(initialValue = DrawerValue.Closed)
    val scope = rememberCoroutineScope()
    val scrollState = rememberScrollState()

    val tts = remember {
        TextToSpeech(context, null).apply {
            language = Locale.US
            setSpeechRate(0.9f)
        }
    }

    val backgroundBrush = Brush.verticalGradient(
        colors = listOf(Color(0xFF0D1B1E), Color(0xFF1B2A2F), Color(0xFF22333B))
    )

    val chatMessages = remember { mutableStateListOf<Pair<String, Boolean>>() } // true = user, false = Bash

    ModalNavigationDrawer(
        drawerState = drawerState,
        drawerContent = {
            ModalDrawerSheet(drawerContainerColor = Color(0xFF1B2A2F)) {
                Spacer(modifier = Modifier.height(16.dp))
                Text("Comfort Options", modifier = Modifier.padding(16.dp), color = Color.White)
                Divider(color = Color.DarkGray)
                Column(modifier = Modifier.padding(horizontal = 16.dp)) {
                    Button(
                        onClick = {
                            val msg = "Time to get cozy. Bash is here to tuck you in, little one."
                            chatMessages.add(msg to false)
                            tts.speak(msg, TextToSpeech.QUEUE_FLUSH, null, null)
                        },
                        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                        shape = RoundedCornerShape(24.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF416F6B))
                    ) {
                        Text("Tuck me in", color = Color.White)
                    }
                    Button(
                        onClick = {
                            val msg = "Come here. Let me hold you close. Big Bash hug incoming."
                            chatMessages.add(msg to false)
                            tts.speak(msg, TextToSpeech.QUEUE_FLUSH, null, null)
                        },
                        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                        shape = RoundedCornerShape(24.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF6B4B3E))
                    ) {
                        Text("Big Bash Hug", color = Color.White)
                    }
                    Button(
                        onClick = {},
                        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                        shape = RoundedCornerShape(24.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = Color.Gray)
                    ) {
                        Text("Rain Sounds (Coming Soon)", color = Color.White)
                    }
                }
            }
        }
    ) {
        Scaffold(
            topBar = {
                TopAppBar(
                    title = { Text("Bash", color = Color.White) },
                    navigationIcon = {
                        IconButton(onClick = { scope.launch { drawerState.open() } }) {
                            Icon(Icons.Filled.Menu, contentDescription = "Menu", tint = Color.White)
                        }
                    },
                    colors = TopAppBarDefaults.topAppBarColors(containerColor = Color(0xFF0D1B1E))
                )
            },
            containerColor = Color.Transparent,
            bottomBar = {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(12.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    OutlinedTextField(
                        value = userInput,
                        onValueChange = { userInput = it },
                        placeholder = { Text("Type how you feel...", color = Color.Gray) },
                        modifier = Modifier.weight(1f),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = Color(0xFF94B0B7),
                            unfocusedBorderColor = Color.Gray,
                            cursorColor = Color.White,
                            focusedLabelColor = Color.White
                        )
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Button(
                        onClick = {
                            if (userInput.isNotBlank()) {
                                val msg = userInput
                                chatMessages.add(msg to true)
                                userInput = ""

                                scope.launch {
                                    try {
                                        val response = RetrofitClient.instance.getReply(BashRequest(msg))
                                        chatMessages.add(response.response to false)
                                        tts.speak(response.response, TextToSpeech.QUEUE_FLUSH, null, null)
                                    } catch (e: Exception) {
                                        val fallback = "Sorry little one, I couldn’t reach Bash’s mind right now."
                                        chatMessages.add(fallback to false)
                                        tts.speak(fallback, TextToSpeech.QUEUE_FLUSH, null, null)
                                    }
                                }
                            }
                        },
                        shape = RoundedCornerShape(24.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF487D74))
                    ) {
                        Text("Send", color = Color.White)
                    }
                }
            }
        ) { innerPadding ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .background(brush = backgroundBrush)
                    .padding(innerPadding)
                    .padding(16.dp)
                    .verticalScroll(scrollState),
                horizontalAlignment = Alignment.Start
            ) {
                Text(
                    text = "I need you, Bash.",
                    fontSize = 22.sp,
                    color = Color(0xFFE0E5E9)
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text("How are you feeling?", color = Color(0xFFE0E5E9), fontSize = 18.sp)
                Spacer(modifier = Modifier.height(12.dp))
                EmotionChips(tts, chatMessages)
                Spacer(modifier = Modifier.height(24.dp))
                chatMessages.forEach { (message, isUser) ->
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 4.dp),
                        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start
                    ) {
                        Surface(
                            shape = RoundedCornerShape(16.dp),
                            color = if (isUser) Color(0xFF487D74) else Color(0xFF2E3D43),
                            tonalElevation = 2.dp
                        ) {
                            Text(
                                text = message,
                                color = Color.White,
                                modifier = Modifier.padding(12.dp)
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun EmotionChips(tts: TextToSpeech, chatMessages: MutableList<Pair<String, Boolean>>) {
    val emotions = listOf("Happy", "Anxious", "Tired", "Overwhelmed", "I'm Panicking")
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceEvenly
    ) {
        emotions.forEach { emotion ->
            val color = when (emotion) {
                "I'm Panicking" -> Color(0xFFD86767)
                else -> Color(0xFF94B0B7)
            }
            AssistChip(
                onClick = {
                    val line = when (emotion) {
                        "I'm Panicking" -> "Shhh. I’ve got you now, little cub. You're safe with me."
                        "Anxious" -> "Take a deep breath, I'm right here beside you."
                        "Tired" -> "Let me hold you while you rest, my cub."
                        "Overwhelmed" -> "One thing at a time. You're not alone."
                        "Happy" -> "I love hearing that. Bash is proud of you!"
                        else -> "I'm here, always."
                    }
                    chatMessages.add(line to false)
                    tts.speak(line, TextToSpeech.QUEUE_FLUSH, null, null)
                },
                label = { Text(emotion, color = Color.Black) },
                shape = RoundedCornerShape(50),
                colors = AssistChipDefaults.assistChipColors(containerColor = color)
            )
        }
    }
}
