import pandas as pd

df = pd.read_csv("D:\mai\dataright.csv", encoding='utf-8-sig')

# Более точные ключевые слова
keywords = [
    'разработка', 'ИТ', 'программное', 'информационная система',
    'веб', 'сайт', 'мобильное приложение', 'ИИ', 'нейросеть',
    'софт', 'приложение'
]

df['is_it'] = df['tender_name'].astype(str).str.lower().str.contains('|'.join(keywords), na=False)

def classify_tender(text):
    text = str(text).lower()
    if 'ии' in text or 'нейросет' in text or 'машинное обучение' in text:
        return 'Проекты с ИИ'
    elif ('чат' in text and 'бот' in text) or ('голосов' in text and 'бот' in text):
        return 'Чат-боты / Голосовые системы'
    elif 'веб' in text or 'сайт' in text:
        return 'Веб-разработка'
    elif 'мобильн' in text or 'приложен' in text:
        return 'Мобильная разработка'
    elif 'erp' in text or 'crm' in text:
        return 'ERP/CRM-системы'
    else:
        return 'Другая ИТ-разработка'

df['category'] = None
df.loc[df['is_it'], 'category'] = df.loc[df['is_it'], 'tender_name'].apply(classify_tender)

print(f"Всего: {df.shape[0]}")
print(f"ИТ-закупок: {df['is_it'].sum()}")
print("Распределение:")
print(df['category'].value_counts(dropna=False))

df.to_csv('FINAL_data.csv', index=False, encoding='utf-8-sig', sep=';')
