import json, click
from agent.graph_hybrid import create_graph

workflow = create_graph()

@click.command()
@click.option('--batch', required=True)
@click.option('--out', default='outputs_hybrid.jsonl')
def run(batch: str, out: str):
    results = []
    with open(batch) as f:
        for line in f:
            q = json.loads(line.strip())
            output = workflow.invoke({"question": q["question"], "question_id": q["id"]})
            results.append(output)
    
    with open(out, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    print(f"Done! Saved {len(results)} answers → {out}")

if __name__ == "__main__":
    run()