(function () {
  const livePanel = document.getElementById("livePanel");
  const chatBox = document.getElementById("chatBox");
  const tickerEl = document.getElementById("ticker");
  const messageEl = document.getElementById("message");
  const sendBtn = document.getElementById("sendBtn");
  const watchInput = document.getElementById("watchTicker");
  const watchBtn = document.getElementById("watchBtn");
  const chipLive = document.getElementById("chipLive");
  const chipChat = document.getElementById("chipChat");

  const wsProto = window.location.protocol === "https:" ? "wss" : "ws";
  const host = window.location.host;

  let liveWs = null;
  let chatWs = null;
  let liveReconnectTimer = null;
  let chatReconnectTimer = null;
  let liveBackoff = 1000;
  let chatBackoff = 1000;

  const latestByTicker = {};
  let activeAssistantBody = null;
  let activeStreamId = null;

  function escapeHtml(s) {
    return String(s || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
  }

  function fmtPrice(n) {
    if (n == null || n === "" || Number.isNaN(Number(n))) return "—";
    const x = Number(n);
    const d = Math.abs(x) >= 100 ? 2 : 2;
    return x.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
  }

  function fmtSentiment(s) {
    if (s == null || s === "" || Number.isNaN(Number(s))) return "—";
    return Number(s).toFixed(3);
  }

  function sentimentClass(score) {
    const n = Number(score);
    if (Number.isNaN(n)) return "neutral";
    if (n > 0.08) return "bull";
    if (n < -0.08) return "bear";
    return "neutral";
  }

  function setChip(el, state, label) {
    if (!el) return;
    el.classList.remove("live", "error", "reconnecting");
    if (state === "live") el.classList.add("live");
    if (state === "error") el.classList.add("error");
    if (state === "reconnecting") el.classList.add("reconnecting");
    const text = el.querySelector(".chip-text");
    if (text) text.textContent = label;
  }

  function sendPong(ws) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "pong", payload: {} }));
    }
  }

  function mergeLiveRow(ticker, patch) {
    const prev = latestByTicker[ticker] || {};
    latestByTicker[ticker] = { ...prev, ...patch };
  }

  function renderLive() {
    const tickers = Object.keys(latestByTicker).sort();
    if (!tickers.length) {
      livePanel.innerHTML =
        '<div class="live-empty"><div class="spinner" aria-hidden="true"></div>' +
        "<div>Waiting for live snapshots from the backend…</div>" +
        '<div style="margin-top:0.5rem;font-size:0.8rem;">Add a symbol below to start streaming that ticker.</div></div>';
      return;
    }

    const cards = tickers
      .map((t) => {
        const d = latestByTicker[t];
        const sent = sentimentClass(d.sentiment_score);
        const pillLabel =
          sent === "bull" ? "Bullish lean" : sent === "bear" ? "Bearish lean" : "Neutral";
        const conf =
          d.model_confidence != null && !Number.isNaN(Number(d.model_confidence))
            ? `${(Number(d.model_confidence) * 100).toFixed(1)}%`
            : "—";
        return (
          '<article class="ticker-card" data-ticker="' +
          escapeHtml(t) +
          '"><div class="ticker-card-top"><span class="ticker-symbol">' +
          escapeHtml(t) +
          '</span><span class="sentiment-pill ' +
          sent +
          '">' +
          pillLabel +
          '</span></div><div class="metrics-grid">' +
          '<div class="metric"><span class="metric-label">Last price</span><span class="metric-value">$' +
          fmtPrice(d.latest_price) +
          '</span></div><div class="metric"><span class="metric-label">Predicted</span><span class="metric-value">$' +
          fmtPrice(d.predicted_price) +
          '</span></div><div class="metric"><span class="metric-label">Model conf.</span><span class="metric-value">' +
          conf +
          '</span></div><div class="metric"><span class="metric-label">Sentiment</span><span class="metric-value">' +
          fmtSentiment(d.sentiment_score) +
          '</span></div><div class="metric"><span class="metric-label">Emotion</span><span class="metric-value">' +
          escapeHtml(d.dominant_emotion ?? "—") +
          '</span></div><div class="metric"><span class="metric-label">Social / news</span><span class="metric-value">ST ' +
          (d.tweets_count ?? 0) +
          " · RD " +
          (d.reddit_count ?? 0) +
          " · N " +
          (d.news_count ?? 0) +
          "</span></div></div></article>"
        );
      })
      .join("");
    livePanel.innerHTML = '<div class="live-cards">' + cards + "</div>";
  }

  function appendChatBubble(role, text, extraClass) {
    const wrap = document.createElement("div");
    wrap.className = "msg msg-" + role + (extraClass ? " " + extraClass : "");
    const roleEl = document.createElement("div");
    roleEl.className = "msg-role";
    roleEl.textContent =
      role === "user" ? "You" : role === "assistant" ? "Advisor" : "System";
    const body = document.createElement("div");
    body.className = "msg-body";
    body.textContent = text || "";
    wrap.appendChild(roleEl);
    wrap.appendChild(body);
    chatBox.appendChild(wrap);
    chatBox.scrollTop = chatBox.scrollHeight;
    return body;
  }

  function connectLive() {
    if (liveWs && liveWs.readyState === WebSocket.OPEN) return;
    setChip(chipLive, "reconnecting", "Live: connecting…");
    liveWs = new WebSocket(wsProto + "://" + host + "/ws/live");

    liveWs.onopen = () => {
      liveBackoff = 1000;
      setChip(chipLive, "live", "Live: connected");
      if (liveReconnectTimer) {
        clearTimeout(liveReconnectTimer);
        liveReconnectTimer = null;
      }
    };

    liveWs.onclose = () => {
      setChip(chipLive, "error", "Live: disconnected");
      liveWs = null;
      scheduleLiveReconnect();
    };

    liveWs.onerror = () => setChip(chipLive, "error", "Live: error");

    liveWs.onmessage = (ev) => {
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (msg.type === "ping") {
        sendPong(liveWs);
        return;
      }
      if (msg.type === "live_connected") return;

      if (msg.type === "live_snapshot" && msg.payload) {
        const t = msg.payload.ticker;
        mergeLiveRow(t, msg.payload.snapshot || {});
        renderLive();
        return;
      }
      if (msg.type === "live_delta" && msg.payload) {
        const t = msg.payload.ticker;
        mergeLiveRow(t, msg.payload.changes || {});
        renderLive();
        return;
      }
      if (msg.type === "stream_tick" && msg.payload) {
        const p = msg.payload;
        const t = p.ticker;
        if (t) {
          mergeLiveRow(t, {
            latest_price: p.price,
            predicted_price: p.prediction,
            sentiment_score: p.sentiment,
          });
          renderLive();
        }
      }
    };
  }

  function scheduleLiveReconnect() {
    if (liveReconnectTimer) return;
    liveReconnectTimer = setTimeout(() => {
      liveReconnectTimer = null;
      setChip(chipLive, "reconnecting", "Live: reconnecting…");
      liveBackoff = Math.min(liveBackoff * 1.5, 30000);
      connectLive();
    }, liveBackoff);
  }

  function connectChat() {
    if (chatWs && chatWs.readyState === WebSocket.OPEN) return;
    setChip(chipChat, "reconnecting", "Chat: connecting…");
    chatWs = new WebSocket(wsProto + "://" + host + "/ws/chat");

    chatWs.onopen = () => {
      chatBackoff = 1000;
      setChip(chipChat, "live", "Chat: connected");
      if (chatReconnectTimer) {
        clearTimeout(chatReconnectTimer);
        chatReconnectTimer = null;
      }
    };

    chatWs.onclose = () => {
      setChip(chipChat, "error", "Chat: disconnected");
      chatWs = null;
      scheduleChatReconnect();
    };

    chatWs.onerror = () => setChip(chipChat, "error", "Chat: error");

    chatWs.onmessage = (ev) => {
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (msg.type === "ping") {
        sendPong(chatWs);
        return;
      }
      if (msg.type === "chat_connected") return;

      const p = msg.payload || {};

      if (msg.type === "chat_start") {
        activeStreamId = p.stream_id;
        activeAssistantBody = appendChatBubble("assistant", "");
        return;
      }
      if (msg.type === "chat_token") {
        if (p.stream_id && p.stream_id !== activeStreamId) return;
        if (!activeAssistantBody) activeAssistantBody = appendChatBubble("assistant", "");
        activeAssistantBody.textContent += p.token || "";
        chatBox.scrollTop = chatBox.scrollHeight;
        return;
      }
      if (msg.type === "chat_end") {
        if (p.stream_id && p.stream_id !== activeStreamId) return;
        activeAssistantBody = null;
        activeStreamId = null;
        return;
      }
      if (msg.type === "error") {
        appendChatBubble("error", p.message || "Unknown error", "msg-error");
        activeAssistantBody = null;
        activeStreamId = null;
      }
    };
  }

  function scheduleChatReconnect() {
    if (chatReconnectTimer) return;
    chatReconnectTimer = setTimeout(() => {
      chatReconnectTimer = null;
      setChip(chipChat, "reconnecting", "Chat: reconnecting…");
      chatBackoff = Math.min(chatBackoff * 1.5, 30000);
      connectChat();
    }, chatBackoff);
  }

  function sendChat() {
    const message = messageEl.value.trim();
    if (!message) return;
    if (!chatWs || chatWs.readyState !== WebSocket.OPEN) {
      appendChatBubble(
        "error",
        "Chat is not connected yet. Wait a moment and try again.",
        "msg-error"
      );
      return;
    }
    appendChatBubble("user", message);
    chatWs.send(
      JSON.stringify({
        type: "chat_message",
        payload: { message, ticker: tickerEl.value.trim().toUpperCase() || "AAPL" },
      })
    );
    messageEl.value = "";
  }

  watchBtn.addEventListener("click", () => {
    const sym = (watchInput.value || "").trim().toUpperCase();
    if (!sym || !liveWs || liveWs.readyState !== WebSocket.OPEN) return;
    liveWs.send(JSON.stringify({ type: "subscribe", payload: { tickers: [sym] } }));
    watchInput.value = "";
  });

  sendBtn.addEventListener("click", sendChat);
  messageEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendChat();
    }
  });

  renderLive();
  connectLive();
  connectChat();
})();
