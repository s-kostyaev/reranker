import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def construct_pairs(query, docs):
    return [[query, doc] for doc in docs]

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')
model.eval()


pairs = construct_pairs('what is panda?', ['hi', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.', 'What a cute little panda', 'Больша́я па́нда, или бамбу́ковый медве́дь[2] (лат. Ailuropoda melanoleuca) — вид всеядных млекопитающих из семейства медвежьих[3][4] (Ursidae) со своеобразной чёрно-белой окраской шерсти, обладающих некоторыми признаками енотов. Единственный современный вид рода Ailuropoda подсемейства Ailuropodinae. Большие панды обитают в горных регионах центрального Китая: Сычуани, на юге Ганьсу и Шэньси. Со второй половины XX века панда стала чем-то вроде национальной эмблемы Китая. Китайское название (кит. упр. 熊猫, пиньинь xióngmāo) означает «медведь-кошка». Его западное имя происходит от малой панды. Раньше его также называли пятни́стым медве́дем (Ursus melanoleucus). Известны случаи нападений больших панд на человека[5][6].', 'Ма́лая па́нда[2], коша́чий медве́дь[3] , кра́сная (рыжая) па́нда или гимала́йский ено́т (лат. Ailurus fulgens) — млекопитающее из семейства пандовых, подотряда псообразных, отряда хищных, которое питается преимущественно растительностью; по размеру примерно соответствует крупным особям домашней кошки.'])
print(pairs)
with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    print(scores)
