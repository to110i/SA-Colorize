import torch
import kornia
import time
import os
import datetime
from torchvision.utils import save_image

from models import Generator, Discriminator


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
        start_time = time.time()

        for step in range(self.total_step):
            self.D.train()
            self.G.train()

            try:
                real_img = next(data_iter)
            except:
                data_iter = iter(self.dataloader)
                real_img = next(data_iter)
            real_Lab = kornia.color.rgb_to_lab(real_img)
            real_Lab = real_Lab.cuda()
            # B x C x H x W
            # dr1, dr2, df1, df2, gf1, gf2 are attention scores
            d_out_real, dr1, dr2 = self.D(real_Lab)
            d_loss_real = self.calc_loss(d_out_real, True)
            real_L = real_Lab[:, 0, :, :]
            real_ab = real_Lab[:, 1:3, :, :]
            fake_ab, gf1, gf2 = self.G(real_L)
            d_out_fake, df1, df2 = self.D(fake_ab)
            d_loss_fake = self.calc_loss(d_out_fake, False)

            d_loss = d_loss_real + d_loss_fake
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            if self.adv_loss == 'wgan-gp':
                alpha = torch.rand(real_ab.size(
                    0), 1, 1, 1).cuda().expand_as(real_Lab)
                fake_Lab = torch.cat([real_L, real_ab], 1)
                interpolated = torch.autograd.Variable(
                    alpha * real_Lab.data + (1 - alpha) * fake_Lab.data, requires_grad=True)
                out, _, _ = self.D(interpolated)

                grad = torch.autograd.grad(
                    outputs=out,
                    inputs=interpolated,
                    grad_outputs=torch.ones(out.size()).cuda(),
                    retain_graph=True,
                    create_graph=True,
                    only_inputs=True)[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

                d_loss = self.lambda_gp * d_loss_gp

                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

            # ============ Train G and gumbel ============ #
            fake_ab, _, _ = self.G(real_L)
            fake_Lab = torch.cat([real_L, real_ab], 1)

            g_out_fake, _, _ = self.D(fake_Lab)  # batch x n
            if self.adv_loss == 'wgan-gp':
                g_loss_fake = - g_out_fake.mean()
            elif self.adv_loss == 'hinge':
                g_loss_fake = - g_out_fake.mean()

            self.reset_grad()
            g_loss_fake.backward()
            self.g_optimizer.step()

            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out_real: {:.4f}, "
                      " ave_gamma_l3: {:.4f}, ave_gamma_l4: {:.4f}".
                      format(elapsed, step + 1, self.total_step, (step + 1),
                             self.total_step, d_loss_real.data[0],
                             self.G.attn1.gamma.mean().data[0], self.G.attn2.gamma.mean().data[0]))

            # Sample images
            if (step + 1) % self.sample_step == 0:
                test_L = real_L[0, :, :, :]
                fake_ab, _, _ = self.G(test_L)
                fake_ab = (fake_ab + 1.0) / 2.0
                fake_Lab = torch.cat([real_L, real_ab])
                # 1 x C x H x W
                fake_images = kornia.color.lab_to_rgb(fake_Lab)
                save_image((fake_images.data),
                           os.path.join(self.sample_path, '{}_fake.png'.format(step + 1)))

            if (step+1) % model_save_step == 0:
                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))
                torch.save(self.D.state_dict(),
                           os.path.join(self.model_save_path, '{}_D.pth'.format(step + 1)))
