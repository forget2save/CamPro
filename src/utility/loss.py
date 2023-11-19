from utils.loss import ComputeLoss

loss_fn = None

def compute_loss(model, images, labels):
    global loss_fn
    if loss_fn is None:
        loss_fn = ComputeLoss(model)
    preds = model(images)
    loss, _ = loss_fn(preds, labels)
    return loss

