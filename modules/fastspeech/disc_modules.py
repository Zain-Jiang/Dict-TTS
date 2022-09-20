import numpy as np
import torch
import torch.nn as nn


class JCU_Discriminator(nn.Module):
    def __init__(self, c_x=80, c_cond=256, c_base=128):
        super(JCU_Discriminator, self).__init__()
        self.cond_conv = nn.Sequential(
            nn.ReflectionPad1d(2),
            nn.utils.weight_norm(nn.Conv1d(c_cond, c_base, kernel_size=5, stride=2)),
            nn.LeakyReLU(0.2, True),
        )
        x_conv = [nn.ReflectionPad1d(2),
                  nn.utils.weight_norm(nn.Conv1d(c_x, c_base, kernel_size=5, stride=2)),
                  nn.LeakyReLU(0.2, True)]
        x_conv += [
            nn.utils.weight_norm(nn.Conv1d(
                c_base,
                c_base,
                kernel_size=5,
                stride=1,
                padding=2,
            )
            ),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.Conv1d(
                c_base,
                c_base,
                kernel_size=5,
                stride=1,
                padding=2,
            )),
            nn.LeakyReLU(0.2),
        ]
        self.x_conv = nn.Sequential(*x_conv)
        self.cond_conv2 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(c_base * 2, c_base, kernel_size=5, stride=1, padding=2)),
            nn.LeakyReLU(0.2, True),
            nn.utils.weight_norm(nn.Conv1d(
                c_base, 1, kernel_size=3, stride=1, padding=1
            ))
        )
        self.x_conv2 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(c_base, c_base, kernel_size=5, stride=1, padding=2)),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(
                c_base, 1, kernel_size=3, stride=1, padding=1
            )
        )

    def forward(self, x, c):
        x = x.transpose(1, 2)
        c = c.transpose(1, 2)
        out = self.cond_conv(c)
        out1 = self.x_conv(x)
        out = torch.cat([out, out1], dim=1)
        cond_out = self.cond_conv2(out)
        uncond_out = self.x_conv2(out1)
        ret = {'y_c': uncond_out, 'y': cond_out}
        return ret
