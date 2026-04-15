from core.model import Model, ModelConfig

config = ModelConfig(n_individuals=5000, n_genes=3)
model = Model(config)
model.run(600)

model_df = model.datacollector.get_model_vars_dataframe()
print("Model stats:")
print(model_df.tail())

agent_df = model.datacollector.get_agent_vars_dataframe()
print("\nAgent stats:")
print(agent_df.describe())
