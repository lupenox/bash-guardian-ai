package com.lupenox.bashguardianai.api

import com.lupenox.bashguardianai.data.BashRequest
import com.lupenox.bashguardianai.data.BashResponse
import retrofit2.http.Body
import retrofit2.http.POST

interface BashService {
    @POST("chat")
    suspend fun getReply(@Body request: BashRequest): BashResponse
}
