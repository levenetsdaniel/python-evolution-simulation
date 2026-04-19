from core.model import Model
from config.sim_config import SimConfig

model = Model(SimConfig())
model.run(600)

model_df = model.datacollector.get_model_vars_dataframe()
print("Model stats:")
print(model_df.tail())

agent_df = model.datacollector.get_agent_vars_dataframe()
print("\nAgent stats:")
print(agent_df.describe())
