from cos_eor.policy.context import ContextModule, PlannerAdapterContextModule, SayPlanContextModule, SayCanContextModule

debug_info = ContextModule.get_debug_info()

graph = debug_info['GRAPH']
print(graph.graph)