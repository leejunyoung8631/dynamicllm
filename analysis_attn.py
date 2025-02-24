import torch
from modelutils import get_model, set_inference

import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def load_model(model_name):
    if model_name.endswith(".bin"):
        pruned_dict = torch.load(model_name)
        tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
        model = model.half()
        generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
    return generator



# path = "/disk/yskim/LLM-Pruner/cellprune_results/llama31_25pruned_more_kd_only/merged/pytorch_model.bin"
path = "/home/tako/ljy/dynamicllm/weight/llama31_25pruned_part75/merged/pytorch_model.bin"
device = "cuda"




# give input here
# user_input = "Using the information contained in the context, give a comprehensive answer to the question.\nIf the answer cannot be deduced from the context, do not give an answer.\nContext:일반 비영리법인과 비교하여 협의에 공익법인에는 상속세 및 증여세법에 따른 혜택이 부여되는 반면, 공익법인법에 따른 엄격한 규제가 이루어지게 된다.\n혜택\n① 출연자에 대한 상속세면제(상증법 제16조 제1항)\n② 공익법인에 대한 증여세면제(상증법 제48조 제2항)\n③ 고유목적준비금의 손금삽입(법인세법 제29조)\n④ 재화 및 용역에 대한 부가가치세 면제(부가가치세법 제12조 제1항) 등\n규제\n① 사업범위가 ｢공익법인법｣ 시행령 제2조의 규정범위 내로 제한\n② ‘이사회’ 설치가 의무(제6조)\n③ 임원 관련 요건이 강화(제5조)\n④ 주무관청의 승인을 받아 상근 임직원 수를 정함(제5조 제9항)\n⑤ 수익사업을 하려면 주무관청의 승인이 필요(제4조 제3항)\n⑥ 기본재산 처분에 주무관청의 허가 필요(제11조 제2항)\n⑦ 잔여재산 귀속이 제한(제13조)\n⑧ 주무관청의 관리감독권이 구체화(제14조 제2항, 제3항)\n⑨ 비영리법인의 경우 과태료 제재만이 있으나, 공익법인은 징역･벌금 등의 형사처벌제재도 존재(제19조)\n- 민법과 공익법인법은 일반법과 특별법의 관계로써, 비영리법인 중에서 공익법인의 설립에 관해서는 특별법 우선적용의 원칙에 따라 공익법인법 제4조 이하의 규정이 민법에 우선해서 적용되고, 해산에 관한 규정 등 공익법에 규정되지 않은 부분은 민법의 규정이 적용된다(공익법인법 제1조 참조).\n==== 페이지 297 ====\n4) 특수법인\n- 특수법인은 사립학교법, 사회복지사업법, 의료법, 변호사법 등 각종 개별법에 따라 설립된 법인을 말한다.\n- 비영리법인 중에서 민법 제32조의 규정에 의하지 않고 각종 개별법에 근거하여 설립된 법인을 통칭하며, 학교법인, 사회복지법인, 의료법인, 법무법인 기타 개별법에 의하여 법인격이 부여된 각종 조합 및 연합회 등이 이에 해당된다.\n- 또한 강학상으로는 한국은행법, 한국도로공사법, 한국연구재단법 등 특수한 공공목적을 수행하기 위해서 특별법에 의해 설립된 법인도 특수법인의 개념에 포함된다. 이를 법정법인으로 부르기도 한다.\n<법인의  분류>\n법인-사단법인-영리법인(회사),\n\nQuestion:공립 학교를 설립하고 운영하기 위해서는 어떤 절차를 밟아야 하나요?"
# user_input = "Using the information contained in the context, give a comprehensive answer to the question.\nIf the answer cannot be deduced from the context, do not give an answer.\nContext:페이지 291 ====\n○ 그러나 단체가 법인격을 취득하여 법인으로 인정되면, 법인과 구성은의 법인격이 서로 구별되므로, 법인의 재산과 구성원 개인의 재산은 엄격히 분리된다.\n참고\n예컨대, 100명의 회원이 회비를 납부하여 조성된 법인의 재산과 각 회원의 개인재산은 엄격히 구분된다. 따라서 법인명의의 사업으로 벌어들인 재산은 법인의 고유재산으로 귀속하게 되고, 법인명의를 부담한 채무는 법인의 재산으로 변제하여야 한다. 법인이 벌어들인 재산을 회원이 개인적으로 착복하게 되면 횡령죄 또는 배임죄가 성립한다. 법인 명의로 부담한 채무에 대해서 회원은 원칙적으로 변제할 의무가 없다.\n\nQuestion:구성원과 법인 사이의 관계는 어떤 특징을 가지고 있나요? (ex. 재산의 분리, 채무의 부담 등)"
user_input =  "Using the information contained in the context, give a comprehensive answer to the question.\nIf the answer cannot be deduced from the context, do not give an answer.\nContext:제공시간|대상자|본인부담금\n월 24시간(A형)|수급자 및 차상위 (가형)|면제\n월 24시간(A형)|중위소득 70%이하 (나형)| 월24,770원\n월 27시간(B형)|수급자 및 차상위 (가형)|월13,930원\n월 27시간(B형)|중위소득 70%이하 (나형)|월27,860원\n월 40시간(C형)|의료급여수급자 중 장기입원 사례관리 퇴원자|면제\n\nQuestion:월 27 시간 B형 혜택을 받을 수 있는 대상자는 누구인가요?"
user_input_org = user_input
history = None
history_internal = None

# Initial system prompt
system_prompt = {"role": "system", "content": "You are a helpful assistant"}

# Initialize history if it's None
if history is None or len(history) == 0:
    history = [system_prompt]
if history_internal is None or len(history_internal) == 0:
    history_internal = [system_prompt]

# Append user input to history
history.append({"role": "user", "content": user_input_org})
history_internal.append({"role": "user", "content": user_input})

history_view = history_internal



generator = load_model(path)
max_length = 512
temperature = 0.7
top_p = 0.9

terminators = [  # TODO: move somewhere? to avoid repeated computation?
            generator.tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
            generator.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]



for i in range(100):
    response = generator(
            history_view,
            #max_length=max_length, #512,
            max_new_tokens=max_length,
            pad_token_id=generator.tokenizer.eos_token_id,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature, #0.7,
            top_p=top_p, #0.9
        )[-1]["generated_text"][-1]["content"]
    
    print(response)
    print("\n")





