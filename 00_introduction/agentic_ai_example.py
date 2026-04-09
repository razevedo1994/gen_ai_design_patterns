from pydantic_ai import Agent
from dataclasses import dataclass

MODEL_ID = ""

@dataclass
class InventoryItem:
    name: str
    quantity_on_hand: int
    weekly_quantity_sold_past_n_weeks: list[int]
    weeks_to_deliver: int

@dataclass
class Reorder:
    name: str
    quantity_to_order: int
    reason_to_order: str

items = [
    InventoryItem("itemA", 300, [50, 70, 80, 100], 2),
    InventoryItem("itemB", 100, [70, 80, 90, 70], 2),
    InventoryItem("itemC", 200, [80, 70, 90, 80], 1)
]

agent = Agent(
    f"anthropic:{MODEL_ID}",
    system_prompt="You are an inventory manager who orders just in time.",
    result_type=list[Reorder],
    )

result = agent.run_sync(f"""
Identify which of these items need to be reordered this week.

**Items**
{items}
""")
print(result.data)