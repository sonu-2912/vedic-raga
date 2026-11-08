"""Deployment abstention policy; class quality comes from validation, not test."""
import json
from .catalog import ROOT

def expression_suggestion(probabilities,checkpoint_sha256=None):
    ordered=sorted(probabilities.items(),key=lambda x:x[1],reverse=True);label,confidence=ordered[0]
    path=ROOT/'reports/expression_policy.json'
    policy=json.loads(path.read_text()) if path.exists() else None
    matched=policy is not None and policy.get('checkpoint_sha256')==checkpoint_sha256
    eligible=matched and label in policy['eligible_classes']
    reliable=bool(eligible and confidence>=policy['minimum_confidence'] and confidence-ordered[1][1]>=policy['minimum_margin'])
    reason='Expression suggestion requires your confirmation.' if reliable else 'Expression unclear: confidence, separation or validation class quality is insufficient. Choose how you feel; neutral must not be read as feeling at ease.'
    return {'suggested_emotion':label if reliable else None,'abstained':not reliable,'needs_confirmation':True,'reason':reason,'policy_source':'Class-specific validation evidence and conservative deployment thresholds; not a new classifier or mood measurement.'}
