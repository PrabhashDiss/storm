from knowledge_storm import STORMWikiRunnerArguments, STORMWikiRunner, STORMWikiLMConfigs
from knowledge_storm.rm import DuckDuckGoSearch
import dspy
from groq import Groq

lm_configs = STORMWikiLMConfigs()
kwargs = {
    'model': 'llama3-70b-8192',
    'api_key': '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',
    'temperature': 1.0,
    'top_p': 0.9,
}
# STORM is a LM system so different components can be powered by different models to reach a good balance between cost and quality.
# For a good practice, choose a cheaper/faster model for `conv_simulator_lm` which is used to split queries, synthesize answers in the conversation.
# Choose a more powerful model for `article_gen_lm` to generate verifiable text with citations.
groq = dspy.GROQ(max_tokens=8000, **kwargs)
lm_configs.set_conv_simulator_lm(groq)
lm_configs.set_question_asker_lm(groq)
lm_configs.set_outline_gen_lm(groq)
lm_configs.set_article_gen_lm(groq)
lm_configs.set_article_polish_lm(groq)
# Check out the STORMWikiRunnerArguments class for more configurations.
engine_args = STORMWikiRunnerArguments(output_dir="outputs")
rm = DuckDuckGoSearch(k=engine_args.search_top_k)
runner = STORMWikiRunner(engine_args, lm_configs, rm)

topic = input('Topic: ')
runner.run(
    topic=topic,
    do_research=True,
    do_generate_outline=True,
    do_generate_article=True,
    do_polish_article=True,
)
runner.post_run()
runner.summary()
