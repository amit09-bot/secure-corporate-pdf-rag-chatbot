import importlib
for name in ['langchain_openai', 'langchain_openai.ChatOpenAI', 'langchain_openai.OpenAIEmbeddings', 'langchain_classic.chains', 'langchain_classic.memory']:
    try:
        mod = importlib.import_module(name)
        print('OK', name)
        if hasattr(mod, '__all__'):
            print('__all__', [x for x in mod.__all__ if 'Chat' in x or 'OpenAI' in x or 'Conversation' in x or 'Memory' in x])
        print('dir', [x for x in dir(mod) if 'Chat' in x or 'OpenAI' in x or 'Conversation' in x or 'Memory' in x][:20])
    except Exception as e:
        print('ERR', name, type(e).__name__, e)
