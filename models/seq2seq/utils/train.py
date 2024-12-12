from utils.preprocessor import time, timeSince

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
                decoder_optimizer, criterion):
    total_loss = 0

    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, (_, _), _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def training(train_dataloader, encoder, decoder, encoder_opt, decoder_opt, criterion_alg,
          n_epochs, print_every=100):
    print_loss_total = 0 
    loss_each_epoch = []
    
    encoder_optimizer = encoder_opt
    decoder_optimizer = decoder_opt
    criterion = criterion_alg

    start = time.time()
    encoder.train()
    decoder.train()
    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        loss_each_epoch.append(loss)

        print_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('(%d %d%%) Time: %s Avg.Loss: %.5f' % (epoch, epoch / n_epochs * 100,
                                        timeSince(start, epoch / n_epochs), print_loss_avg))

    return loss_each_epoch