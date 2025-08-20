def _should_stop_dynamic(state: InterviewState) -> bool:
    """count==0인 경우에만 호출: 충분히 평가 완료면 True."""
    seq = int(getattr(state, "seq", 0) or 1)
    
    print(f'♥{seq}')
    if seq > DYN_HARD_CAP:
        return True
    if seq >= DYN_MIN_SEQ:
        return True

    last_q = (state.question[-1] if getattr(state, "question", None) else "") or ""
    ans = state.last_answer or (state.answer[-1] if state.answer else "") or ""
    la = getattr(state, "last_analysis", {}) or {}
    good, bad, score = la.get("good", ""), la.get("bad", ""), la.get("score", 0)

    if getattr(state, "language", "KOREAN") == "ENGLISH":
        sys_msg = (
            'Decide whether to end the interview now. '
            'You must output exactly {{"stop": true}} or {{"stop": false}}, only one of the two. '
            'Do not include any other text, explanations, quotes, or comments.'
        )
        user_msg = (
            "last_question: {q}\nlast_answer: {a}\nanalysis.good: {g}\nanalysis.bad: {b}\nscore: {s}\n"
            "Return ONLY JSON."
        )
    else:
        sys_msg = (
            '면접을 지금 종료할지 결정하라. '
            '출력은 반드시 정확히 {{"stop": true}} 또는 {{"stop": false}} 둘 중 하나만. '
            '그 외 다른 텍스트, 설명, 따옴표, 주석을 절대 쓰지 말라.'
        )
        user_msg = (
            "마지막_질문: {q}\n마지막_답변: {a}\n분석.잘한점: {g}\n분석.개선점: {b}\n점수: {s}\n"
            "JSON만 반환."
        )
    print("🔥 sys_msg 원본 =", repr(sys_msg))
    print("🔥 user_msg 원본 =", repr(user_msg))

    p = ChatPromptTemplate.from_messages([("system", sys_msg), ("user", user_msg)])
    print("🔥 p.input_variables =", p.input_variables)
    try:
        p = ChatPromptTemplate.from_messages([("system", sys_msg), ("user", user_msg)])
        resp = (p | llm.bind(max_tokens=12, temperature=0)).invoke({
            "q": last_q[:300], "a": ans[:300], "g": str(good)[:200], "b": str(bad)[:200], "s": score,
        })
        raw = (getattr(resp, "content", str(resp)) or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw) if raw.startswith("{") else {}
        return bool(data.get("stop", False))
    except Exception as e:
        print("⚠️ [동적 종료 판단 실패 → 계속 진행]:", e)
        return False