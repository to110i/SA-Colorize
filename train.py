import torch


class Trainer(object):
    def __init__(self, dataloader, config) -> None:
        self.dataloader = dataloader

    def calc_loss(self, x, real_flag) -> torch.Tensor:
        if real_flag is True:
            x = -x
        if self.adv_loss == 'wgan-gp':
            loss = torch.mean(x)
        if self.adv_loss == 'hinge':
            loss = torch.nn.ReLU()(1.0 + x).mean()
            # ここ1.0で大丈夫...？

    def train(self) -> None:
        data_iter = iter(self.dataloader)

        for epoch in range(epochs):
            self.D.train()
            self.G.train()

            try:
                real_L, real_ab = next(data_iter)

            except:
                data_iter = iter(self.dataloader)
                real_L, real_ab = next(data_iter)
            # real_images = tensor2var(real_images) 必要？？

            # dr1, dr2, df1, df2, gf1, gf2 are attention scores
            d_out_real, dr1, dr2 = self.D(real_ab)
            d_loss_real = self.calc_loss(d_out_real, True)

            fake_ab, gf1, gf2 = self.G(real_L)
            d_out_fake, df1, df2 = self.D(fake_ab)
            d_loss_fake = self.calc_loss(d_out_fake, False)

            d_loss = d_loss_real + d_loss_fake
