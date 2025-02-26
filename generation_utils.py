# Yeseong, CELL, 2025
# Generation utils

# Deepseek tokens
THINK_START_TOKEN = "<think>"
THINK_END_TOKEN = "</think>"
BOS_TOKEN = "<｜begin▁of▁sentence｜>"
EOS_TOKEN = "<｜end▁of▁sentence｜>"
USER_TOKEN = "<｜User｜>"
ASSITANT_TOKEN = "<｜Assistant｜>"


def generate(model, tokenizer, input_text, device, temperature=0.7, is_deepseek=False):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    is_exceed = False
    if len(inputs["input_ids"][0]) >= 2048:
        is_exceed = True
        inputs["input_ids"] = inputs["input_ids"][:, -2048:]  # Remove oldest tokens
        inputs["attention_mask"] = inputs["attention_mask"][:, -2048:]  # Adjust mask

    if not is_deepseek:
        # For 'Bllossom/llama-3.2-Korean-Bllossom-3B'
        terminators = [  # TODO: move somewhere? to avoid repeated computation?
                tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
    else:
        terminators = tokenizer.eos_token_id
    
    output = model.generate(
        **inputs,
        #max_length=2048,  # Maximum length of generated text
        max_new_tokens=2048,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=terminators,
        #num_return_sequences=1,  # Number of sequences to return
        do_sample=True,  # Enable sampling
        #top_k=50,  # Top-k sampling
        top_p=0.9,  # Nucleus sampling
        temperature=temperature,  # Sampling temperature
    )
    
    print("i exited here")
    exit()

    # Decode the output
    input_token_len = inputs.input_ids.shape[-1]
    generated_text = tokenizer.decode(output[0][input_token_len:], skip_special_tokens=True)

    if is_exceed:
        generated_text = "[Context Window Exceed] " + generated_text
    return generated_text


def generate_with_instruction(model, tokenizer, inst, device, temperature=0.7):
    if not inst.startswith("<|start_header_id|>"):
        input_text = tokenizer.apply_chat_template(
                    [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": inst}
                    ],
                    tokenize=False,
                    add_generation_prompt=True  # TODO: Do we need?
                )
    else:
        input_text = inst

    return generate(model, tokenizer, input_text, device, temperature)


def generate_with_instruction_deepseek(model, tokenizer, inst, device, temperature=0.7):
    if inst.startswith("<|start_header_id|>"):
        raise NotImplementedError

    input_text = f"{USER_TOKEN}{inst}\n{ASSITANT_TOKEN}"
    return generate(model, tokenizer, input_text, device, temperature, True)


def generation_test(model, tokenizer, device):
    raw_input_texts = ["안녕하세요, 오늘 날씨는",]
    instruct_texts = ["안녕하세요",
            """Using the information contained in the context, give a comprehensive answer to the question.
If the answer cannot be deduced from the context, do not give an answer.
Context:위해 직원 간의 휴무일은 사전에 조정하여 협의한다.
제7장 업무수행
제29조(업무수행) ① 모든 업무는 시설장의 결재를 받아 집행한다.
② 업무집행 관   지침을 작성하여 직원이 이를 숙지하고 준수하도록 한다.
제30조(아동관리) ① 반드시 주1회 이상의 직원회의를 하여 직원들이 아동들의 정보를 상호교환하며 긴밀히 협조하여 아동지도에 최선을 다해야 한다.
② 시설에 반드시 종사자가 상주하여 아동들만 있게 하지 않는다.
③ 시설 이용 아동의 모든 외부활동은 시설장의 관리 하에 이루어진다.
④ 아동보호를 위한 가정방문, 자원봉사자 및 후원자 방문, 유관기관 방문 등의 외부방문은 반드시 시설장의 관리 하에 이루어진다.
⑤ 아동의 출석상황과 가정한경 변화 등을 정기적으로 점검으로 시설에 잘 적응하고 생활할 수 있도록 한다.
제31조(아동지도) ① 아동을 안전하게 보호하고, 아동은 신체적 정신적으로 건강하게 자랄 수 있는 아동복지서비스를 제공하도록 노력한다.
② 관찰을 통해 아동들의 의사소통 수준을 이해한다.
③ 종사자들의 언어표현과 행동을 아이에 맞는 수준으로 수정한다.
④ 아동 각자의 능력을 고려하여 역할을 부여한다.
⑤ 아동들에게 고운말과 올바른 행동을 하도록 지도하고 종사자가 모범을 보인다.
⑥ 아동의 개별상담, 가족상담, 각 프로그램 기획과 관련한 상담과 사회복지서비스를 제공한다.
제32조(아동훈육) ① 아동을 훈육하는데 있어서 아동의 권익보호를 위한 아동훈육지침을 마련하여 실천하고, 아동을 훈육하는데 있어서 일관성과 형평성이 있어야 한다.
② 문제행동이 나타난 직후에 바로 훈육하도록 하며, 문제해결을 위한 가장 효과적인
==== 페이지 270 ====
훈육방법을 습득하여 아동의 행동을 변화시킬 수 있도록 해야 한다.
③ 어떠한 형태의 체벌(신체적, 정신적)도 아동훈육의 방법으로 이용해서는 안된다.
제33조(위생 및 생활지도) ① 식사 전이나 방과 후등 필요한 경우에는 반드시 손을 씻어야 함을 훈련시키고 실천할 수 있도록 일관성 있게 지도한다.
② 평상시에는 비상약품(소독약, 연고, 밴드, 붕대 등)을

Question:종사자는 아동에게 고운 말을 하도록 가르쳐 주고, 자신도 좋은 행동을 보여주어야 하나?""",
            """Using the information contained in the context, give a comprehensive answer to the question.
If the answer cannot be deduced from the context, do not give an answer.

Context: 빈대는 주로 침구류/매트리스 등에 서식하며 사람을 흡혈 하는 해충으로, 침대에서 주로 서식하여 영어로는 Bed Bug로 불립니다. 최근 해외여행의 증가로 전 세계에 빠르게 확산되고 있으며, 가방과 옷을 통해 널리 전파되고 있습니다. 빈대는 실내기온 (18~20℃)에서 9~18개월 생존하며, 최저온도(13℃이하)와 최고 온도 (45℃ 이상)에서 발육이 중지 됩니다. 주간에는 가구나 침실 벽의 틈에 서식하며 커튼 혹은 벽지 등에 숨어있고 야간에는 흡혈 활동을 하며 저녁보다는 이른 새벽에 더 활발합니다. 주로 온혈 동물의 피를 먹이로 삼고 암수 모두 흡혈하며 주 1-2회에 걸쳐 약 10분간 몸무게의 2.5~6배의 피를 섭취합니다. 빈대는 중고품 가구, 낡은 책, 옷, 여행가방, 박스나 신발을 통해 유입되고 집에서 기르는 반려동물의 몸을 통해서도 발견됩니다. 빈대는 먹이를 섭취하지 않고 18개월 동안 버틸 수 있으며, 1일 최대 30미터 이상 이동할 수 있어 실내 이동만으로 하루 동안 넓은 범위로 확산이 가능합니다.  빈대는 방제 난이도가 가장 높은 해충으로 한번 발생하면 퇴치가 어렵기 때문에, 정기적으로 서식하고 있는지 모니터링을 진행하여 초기에 대응하는 것이 가장 최선의 방법입니다. 

Question: 빈대는 영어로 뭐라고 하나요?""",
            """Using the information contained in the context, give a comprehensive answer to the question.
If the answer cannot be deduced from the context, do not give an answer.
Context:구분|구분|지원내용
시설|요양시설(공동생활가정)|
입소 수급자에게 신체활동 지원 및 심신기능의 유지·향상을 위한 교육·훈련 제공
재가|방문요양|신체활동 및 가사활동 지원
재가|방문목욕|수급자 가정 방문을 통한 목욕 제공
재가|방문간호|
간호사 등이 수급자 가정을 방문하여 간호, 진료보조, 요양에 관한 상담 등 제공
재가|주·야간보호|하루 중 일정 시간 장기요양기관 보호
재가|단기보호|월 9일 이내 장기요양기관에 보호
재가|기타 재가급여(복지용구)|일상생활, 신체활동에필요한 용구 제공

Question:방문 목욕 서비스와 방문 간호 서비스는 어떤 차이점이 있나요?"""
    ]

    for text in raw_input_texts:
        generated_text = generate(model, tokenizer, text, device)
        #print(text)
        print(generated_text)

    for text in instruct_texts:
        generated_text = generate_with_instruction(model, tokenizer, text, device)
        #print(text)
        print(generated_text)