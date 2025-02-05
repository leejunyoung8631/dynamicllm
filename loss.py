import torch
import torch.nn.functional as F

def distillation_loss(student_loss, student_logits, teacher_logits, labels, alpha=0.5, T=2.0):
    ce_loss = student_loss
    
    teacher_probs = F.softmax(teacher_logits / T, dim=1)
    
    student_log_probs = F.log_softmax(student_logits / T, dim=1)

    distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T * T)

    total_loss = alpha * ce_loss + (1 - alpha) * distill_loss
    return total_loss