from collections import Counter

def simple_voting_modes(predictions_per_input_doc, doc_to_choice, task_def):

    aggregated_metrics = {}

    for strategy, top_k in [("majority", None), ("logits", None), ("condorcet", None), ("borda", None), ("rrf", None)]:
        print(f"[Voting Info] Aggregating with simple voting strategy: {strategy}" + (f" (top-{top_k})" if top_k is not None else ""))

        strategy_name = f"{strategy}" + (f"_top{top_k}" if top_k is not None else "")
        results_per_doc = None

        match strategy:
            case "majority":
                results_per_doc = majority_aggregate_votes(predictions_per_input_doc, doc_to_choice, task_def, k=top_k, strategy_name=strategy_name)

            case "logits":
                results_per_doc = logits_aggregate_votes(predictions_per_input_doc, doc_to_choice, task_def, k=top_k, strategy_name=strategy_name)

            case "condorcet":
                results_per_doc = condorcet_aggregate_votes(
                    predictions_per_input_doc, doc_to_choice, task_def,
                    k=top_k, strategy_name=strategy_name
                )

            case "borda":
                results_per_doc = borda_aggregate_votes(
                    predictions_per_input_doc, doc_to_choice, task_def,
                    k=top_k, strategy_name=strategy_name
                )

            case "rrf":
                results_per_doc = rrf_aggregate_votes(
                    predictions_per_input_doc, doc_to_choice, task_def,
                    k=top_k, strategy_name=strategy_name
                )

            case _:
                raise ValueError(f"Unknown strategy {strategy}")

        aggregated_metrics[strategy_name] = aggregate_metrics_per_strategy(results_per_doc, task_def)

    return aggregated_metrics
        
def aggregate_metrics_per_strategy(results_per_doc, task_def):
    aggregated_metrics = {}
    for metric_name, agg_fn in task_def.aggregation().items():
        all_values = [results_per_doc[doc_id][metric_name] for doc_id in results_per_doc if metric_name in results_per_doc[doc_id]]
        if all_values:
            aggregated_metrics[metric_name] = agg_fn(all_values)
    return aggregated_metrics

def majority_vote(predictions: list) -> int:
    """Aggregate predictions using majority voting and return the winning index."""
    counter = Counter(predictions)
    return counter.most_common(1)[0][0]

def majority_aggregate_votes(predictions_per_input_doc, doc_to_choice, task_def, k=None, strategy_name="majority"):
    results_per_doc_id = {}
    for doc_id, info in predictions_per_input_doc.items():
        top_k_preds = [p for p in (info["preds"][:k] if k is not None else info["preds"])]
        pred = majority_vote(top_k_preds)
        predictions_per_input_doc[doc_id][strategy_name] = doc_to_choice[pred]
        pred_probs = [0.0 for _ in doc_to_choice]

        # Aggrate probabilities by averaging them but only from the majority voted class
        for p, probs in zip(info["preds"][:k], info["pred_probs"][:k]):
            if p == pred:
                for i in range(len(doc_to_choice)):
                    pred_probs[i] += probs[i]
        pred_probs = [(p / top_k_preds.count(pred), False) for p in pred_probs]
        results_per_doc_id[doc_id] = task_def.process_results(info["doc"], pred_probs)
    return results_per_doc_id

def logits_aggregate_votes(predictions_per_input_doc, doc_to_choice, task_def, k=None, strategy_name="logits"):
    results_per_doc_id = {}
    for doc_id, info in predictions_per_input_doc.items():
        # Sum logits across all predictions
        summed_logits = [0.0 for _ in doc_to_choice]

        for probs in info["pred_probs"][:k]:
            for i in range(len(doc_to_choice)):
                summed_logits[i] += probs[i]

        pred = summed_logits.index(max(summed_logits))

        predictions_per_input_doc[doc_id]["logits"] = doc_to_choice[pred]

        pred_probs = [(p / len(info["pred_probs"]), False) for p in summed_logits]

        results_per_doc_id[doc_id] = task_def.process_results(info["doc"], pred_probs)
    return results_per_doc_id

def condorcet_aggregate_votes(predictions_per_input_doc, doc_to_choice, task_def, k=None, strategy_name="condorcet"):
    """
    Implements Condorcet voting:
    - Each candidate compared pairwise against others
    - Winner must beat every other candidate in head-to-head matchups
    - If no winner exists, fallback to majority voting
    """
    results_per_doc_id = {}

    for doc_id, info in predictions_per_input_doc.items():
        votes = info["preds"][:k] if k is not None else info["preds"]

        # Count pairwise wins
        num_classes = len(doc_to_choice)
        pairwise_wins = {c: 0 for c in range(num_classes)}
        pairwise_matches = {c: 0 for c in range(num_classes)}

        # Count pairwise outcomes
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                votes_i = sum(1 for v in votes if v == i)
                votes_j = sum(1 for v in votes if v == j)
                if votes_i > votes_j:
                    pairwise_wins[i] += 1
                elif votes_j > votes_i:
                    pairwise_wins[j] += 1
                # Every comparison counts as a match
                pairwise_matches[i] += 1
                pairwise_matches[j] += 1

        # Check for Condorcet winner
        condorcet_winner = None
        for c, wins in pairwise_wins.items():
            if wins == num_classes - 1:  # beat all others
                condorcet_winner = c
                break

        if condorcet_winner is None:
            # Fallback *per-doc*, not return for all docs
            majority_results = majority_aggregate_votes(
                {doc_id: info}, doc_to_choice, task_def, k=k, strategy_name="condorcet_fallback"
            )
            results_per_doc_id.update(majority_results)
            continue

        # Build probability distribution from pairwise win ratio
        scores = [
            pairwise_wins[c] / pairwise_matches[c] if pairwise_matches[c] > 0 else 0.0
            for c in range(num_classes)
        ]

        # Normalize scores to sum=1
        total_score = sum(scores)
        if total_score == 0:
            probs = [1.0 / num_classes] * num_classes  # fallback uniform
        else:
            probs = [s / total_score for s in scores]

        # Convert into (prob, is_predicted) tuples
        pred_probs = [(p, (i == condorcet_winner)) for i, p in enumerate(probs)]

        predictions_per_input_doc[doc_id][strategy_name] = doc_to_choice[condorcet_winner]
        results_per_doc_id[doc_id] = task_def.process_results(info["doc"], pred_probs)

    return results_per_doc_id

def borda_aggregate_votes(predictions_per_input_doc, doc_to_choice, task_def, k=None, strategy_name="borda"):
    """
    Implements Borda count aggregation:
    - Each prediction provides a ranking of classes based on probs
    - Assign points = (num_classes - rank)
    - Sum points across all predictions
    - Highest scoring candidate wins
    """
    results_per_doc_id = {}

    for doc_id, info in predictions_per_input_doc.items():
        num_classes = len(doc_to_choice)
        scores = [0.0 for _ in range(num_classes)]

        # Use top_k predictions if specified
        pred_probs_list = info["pred_probs"][:k] if k is not None else info["pred_probs"]

        for probs in pred_probs_list:
            # Sort classes by probability (descending)
            ranked = sorted(range(num_classes), key=lambda i: probs[i], reverse=True)
            for rank, cls in enumerate(ranked):
                # Higher rank → more points (num_classes - rank - 1)
                scores[cls] += num_classes - rank - 1

        # Winner = argmax of scores
        winner = scores.index(max(scores))

        # Normalize scores to probability-like distribution
        total_score = sum(scores)
        probs = [s / total_score if total_score > 0 else 1.0 / num_classes for s in scores]

        # Convert into (prob, is_predicted) tuples
        pred_probs = [(p, (i == winner)) for i, p in enumerate(probs)]

        predictions_per_input_doc[doc_id][strategy_name] = doc_to_choice[winner]
        results_per_doc_id[doc_id] = task_def.process_results(info["doc"], pred_probs)

    return results_per_doc_id

def rrf_aggregate_votes(predictions_per_input_doc, doc_to_choice, task_def, k=None, strategy_name="rrf", rrf_k=60):
    """
    Implements Reciprocal Rank Fusion (RRF) aggregation:
    - Each ranked list contributes 1 / (rrf_k + rank) to a class score
    - Sum across all predictions
    - Winner = class with highest fused score
    """
    results_per_doc_id = {}

    for doc_id, info in predictions_per_input_doc.items():
        num_classes = len(doc_to_choice)
        scores = [0.0 for _ in range(num_classes)]

        # Use top_k predictions if specified
        pred_probs_list = info["pred_probs"][:k] if k is not None else info["pred_probs"]

        for probs in pred_probs_list:
            # Sort classes by probability (descending)
            ranked = sorted(range(num_classes), key=lambda i: probs[i], reverse=True)
            for rank, cls in enumerate(ranked):
                scores[cls] += 1.0 / (rrf_k + rank + 1)  # rank is 0-based → use (rank+1)

        # Winner = argmax of scores
        winner = scores.index(max(scores))

        # Normalize scores into probability-like distribution
        total_score = sum(scores)
        probs = [s / total_score if total_score > 0 else 1.0 / num_classes for s in scores]

        # Convert into (prob, is_predicted) tuples
        pred_probs = [(p, (i == winner)) for i, p in enumerate(probs)]

        predictions_per_input_doc[doc_id][strategy_name] = doc_to_choice[winner]
        results_per_doc_id[doc_id] = task_def.process_results(info["doc"], pred_probs)

    return results_per_doc_id