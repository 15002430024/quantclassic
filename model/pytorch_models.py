"""
PyTorch Models - PyTorch 深度学习模型实现

提供常用的时序预测模型

🆕 重构说明 (2026-01):
- fit() 方法已代理到 model/train/SimpleTrainer
- 保持原有接口兼容，内部使用统一训练引擎
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
from tqdm import tqdm

from .base_model import PyTorchModel
from .model_factory import register_model


# ==================== LSTM 模型 ====================

class LSTMNet(nn.Module):
    """LSTM 神经网络"""
    
    def __init__(self, d_feat: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, return_hidden: bool = False):
        """
        前向传播
        
        Args:
            x: [batch, seq_len, features] 输入序列
            return_hidden: 是否返回隐藏特征（用于相关性正则化）
            
        Returns:
            如果 return_hidden=False: 预测值 [batch]
            如果 return_hidden=True: (预测值, 隐藏特征) 元组
        """
        # x: [batch, seq_len, features]
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        last_out = lstm_out[:, -1, :]  # [batch, hidden_size]
        
        if return_hidden:
            # 返回元组：(预测值, 隐藏特征)
            out = self.dropout(last_out)
            pred = self.fc(out).squeeze(-1)
            return pred, last_out  # 返回未经过fc前的隐藏特征
        else:
            out = self.dropout(last_out)
            return self.fc(out).squeeze(-1)


@register_model('lstm')
@register_model('LSTM')
class LSTMModel(PyTorchModel):
    """
    LSTM 时序预测模型
    
    Example:
        model = LSTMModel(
            d_feat=20,
            hidden_size=64,
            num_layers=2,
            dropout=0.1,
            n_epochs=100,
            lr=0.001
        )
        model.fit(train_loader, valid_loader)
        predictions = model.predict(test_loader)
    """
    
    def __init__(
        self,
        d_feat: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        Args:
            d_feat: 特征维度
            hidden_size: LSTM 隐藏层大小
            num_layers: LSTM 层数
            dropout: Dropout 概率
        """
        super().__init__(**kwargs)
        
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # 创建模型
        self.model = LSTMNet(d_feat, hidden_size, num_layers, dropout).to(self.device)
        
        # 创建优化器和损失函数
        self.optimizer = self._get_optimizer()
        self.criterion = self._get_loss_fn()
        
        self.logger.info(f"LSTM 模型参数:")
        self.logger.info(f"  特征维度: {d_feat}")
        self.logger.info(f"  隐藏层大小: {hidden_size}")
        self.logger.info(f"  层数: {num_layers}")
        self.logger.info(f"  Dropout: {dropout}")
    
    def fit(self, train_loader, valid_loader=None, save_path: Optional[str] = None):
        """
        训练模型
        
        🆕 重构: 代理到 SimpleTrainer，保持接口兼容
        
        Args:
            train_loader: 训练数据加载器
            valid_loader: 验证数据加载器
            save_path: 模型保存路径
            
        Returns:
            训练结果字典（包含 train_losses, valid_losses, best_epoch 等）
        """
        self.logger.info("开始训练 LSTM 模型...")
        
        # 🆕 使用 SimpleTrainer 代理训练
        from .train import SimpleTrainer, TrainerConfig
        
        trainer_config = TrainerConfig(
            n_epochs=self.n_epochs,
            lr=self.lr,
            early_stop=self.early_stop,
            optimizer=self.optimizer_name,
            loss_fn=self.loss_fn_name,
            loss_kwargs=self.loss_kwargs,
            use_scheduler=self.use_scheduler,
            scheduler_type=self.scheduler_type,
            scheduler_patience=self.scheduler_patience,
            scheduler_factor=self.scheduler_factor,
            scheduler_min_lr=self.scheduler_min_lr,
            lambda_corr=self.lambda_corr,
            checkpoint_dir=str(Path(save_path).parent) if save_path else None,
            save_best_only=True,
        )
        
        trainer = SimpleTrainer(self.model, trainer_config, str(self.device))
        result = trainer.train(train_loader, valid_loader, save_path=save_path)
        
        # 同步训练状态到当前实例
        self.train_losses = result['train_losses']
        self.valid_losses = result['valid_losses']
        self.lr_history = result['lr_history']
        self.best_score = -result['best_score']  # SimpleTrainer 存的是 loss，转为负
        self.best_epoch = result['best_epoch']
        self.fitted = True
        
        self.logger.info(f"训练完成! 最佳 epoch: {self.best_epoch + 1}")
        
        return result
    
    # 🆕 predict() 方法已移至基类 PyTorchModel（2026-01-11 重构）
    # 如需自定义前向逻辑，可覆写 _forward_for_predict() 钩子


# ==================== GRU 模型 ====================

class GRUNet(nn.Module):
    """GRU 神经网络"""
    
    def __init__(self, d_feat: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, return_hidden: bool = False):
        """
        前向传播
        
        Args:
            x: [batch, seq_len, features] 输入序列
            return_hidden: 是否返回隐藏特征（用于相关性正则化）
            
        Returns:
            如果 return_hidden=False: 预测值 [batch]
            如果 return_hidden=True: (预测值, 隐藏特征) 元组
        """
        gru_out, _ = self.gru(x)
        last_out = gru_out[:, -1, :]  # [batch, hidden_size]
        
        if return_hidden:
            out = self.dropout(last_out)
            pred = self.fc(out).squeeze(-1)
            return pred, last_out
        else:
            out = self.dropout(last_out)
            return self.fc(out).squeeze(-1)


@register_model('gru')
@register_model('GRU')
class GRUModel(PyTorchModel):
    """
    GRU 时序预测模型
    
    类似 LSTM，但参数更少，训练更快
    """
    
    def __init__(
        self,
        d_feat: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        self.model = GRUNet(d_feat, hidden_size, num_layers, dropout).to(self.device)
        self.optimizer = self._get_optimizer()
        self.criterion = self._get_loss_fn()
        
        self.logger.info(f"GRU 模型参数:")
        self.logger.info(f"  特征维度: {d_feat}")
        self.logger.info(f"  隐藏层大小: {hidden_size}")
        self.logger.info(f"  层数: {num_layers}")
    
    def fit(self, train_loader, valid_loader=None, save_path: Optional[str] = None):
        """
        训练模型
        
        🆕 重构: 代理到 SimpleTrainer，保持接口兼容
        
        Args:
            train_loader: 训练数据加载器
            valid_loader: 验证数据加载器
            save_path: 模型保存路径
            
        Returns:
            训练结果字典
        """
        self.logger.info("开始训练 GRU 模型...")
        
        # 🆕 使用 SimpleTrainer 代理训练
        from .train import SimpleTrainer, TrainerConfig
        
        trainer_config = TrainerConfig(
            n_epochs=self.n_epochs,
            lr=self.lr,
            early_stop=self.early_stop,
            optimizer=self.optimizer_name,
            loss_fn=self.loss_fn_name,
            loss_kwargs=self.loss_kwargs,
            use_scheduler=self.use_scheduler,
            scheduler_type=self.scheduler_type,
            scheduler_patience=self.scheduler_patience,
            scheduler_factor=self.scheduler_factor,
            scheduler_min_lr=self.scheduler_min_lr,
            lambda_corr=self.lambda_corr,
            checkpoint_dir=str(Path(save_path).parent) if save_path else None,
            save_best_only=True,
        )
        
        trainer = SimpleTrainer(self.model, trainer_config, str(self.device))
        result = trainer.train(train_loader, valid_loader, save_path=save_path)
        
        # 同步训练状态
        self.train_losses = result['train_losses']
        self.valid_losses = result['valid_losses']
        self.lr_history = result['lr_history']
        self.best_score = -result['best_score']
        self.best_epoch = result['best_epoch']
        self.fitted = True
        
        self.logger.info(f"训练完成! 最佳 epoch: {self.best_epoch + 1}")
        
        return result
    
    # 🆕 predict() 方法已移至基类 PyTorchModel（2026-01-11 重构）
    # 修复：旧版 `for batch_x, _` 无法处理图/日级 loader 的 unpack 错误


# ==================== Transformer 模型 ====================

class TransformerNet(nn.Module):
    """Transformer 神经网络"""
    
    def __init__(
        self,
        d_feat: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float
    ):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Linear(d_feat, d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, return_hidden: bool = False):
        """
        前向传播
        
        Args:
            x: [batch, seq_len, features] 输入序列
            return_hidden: 是否返回隐藏特征（用于相关性正则化）
            
        Returns:
            如果 return_hidden=False: 预测值 [batch]
            如果 return_hidden=True: (预测值, 隐藏特征) 元组
        """
        # x: [batch, seq_len, features]
        x = self.input_proj(x)
        x = self.transformer(x)
        # 取最后一个时间步
        x = x[:, -1, :]  # [batch, d_model]
        
        if return_hidden:
            x_hidden = x.clone()  # 保存dropout前的隐藏特征
            x = self.dropout(x)
            pred = self.fc(x).squeeze(-1)
            return pred, x_hidden
        else:
            x = self.dropout(x)
            return self.fc(x).squeeze(-1)


@register_model('transformer')
@register_model('Transformer')
class TransformerModel(PyTorchModel):
    """
    Transformer 时序预测模型
    
    使用自注意力机制捕捉长期依赖
    """
    
    def __init__(
        self,
        d_feat: int = 20,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        Args:
            d_feat: 特征维度
            d_model: Transformer 隐藏维度
            nhead: 注意力头数
            num_layers: Transformer 层数
            dropout: Dropout 概率
        """
        super().__init__(**kwargs)
        
        self.d_feat = d_feat
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        self.model = TransformerNet(
            d_feat, d_model, nhead, num_layers, dropout
        ).to(self.device)
        
        self.optimizer = self._get_optimizer()
        self.criterion = self._get_loss_fn()
        
        self.logger.info(f"Transformer 模型参数:")
        self.logger.info(f"  特征维度: {d_feat}")
        self.logger.info(f"  隐藏维度: {d_model}")
        self.logger.info(f"  注意力头数: {nhead}")
        self.logger.info(f"  层数: {num_layers}")
    
    def fit(self, train_loader, valid_loader=None, save_path: Optional[str] = None):
        """训练模型"""
        self.logger.info("开始训练 Transformer 模型...")
        
        # 🆕 创建学习率调度器
        self.scheduler = self._get_scheduler()
        
        best_valid_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.n_epochs):
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            if valid_loader is not None:
                valid_loss = self._valid_epoch(valid_loader)
                self.valid_losses.append(valid_loss)
                
                self.logger.info(
                    f"Epoch {epoch+1}/{self.n_epochs} - "
                    f"Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}"
                )
                
                # 🆕 调用学习率调度器
                self._step_scheduler(valid_loss)
                
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    self.best_score = -valid_loss
                    self.best_epoch = epoch
                    patience_counter = 0
                    
                    if save_path:
                        self.save_model(save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stop:
                        self.logger.info(f"早停触发 at epoch {epoch+1}")
                        break
            else:
                self.logger.info(f"Epoch {epoch+1}/{self.n_epochs} - Train Loss: {train_loss:.6f}")
                self._step_scheduler(train_loss)
                self.best_epoch = epoch
        
        self.fitted = True
        if valid_loader is not None:
            self.logger.info(f"训练完成! 最佳 epoch: {self.best_epoch+1}")
        else:
            self.logger.info(f"训练完成! (无验证集，共 {self.n_epochs} epochs)")
    
    # 🆕 predict() 方法已移至基类 PyTorchModel（2026-01-11 重构）


# ==================== VAE 模型 ====================

class VAENet(nn.Module):
    """VAE (Variational Autoencoder) 神经网络"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        window_size: int,
        dropout: float
    ):
        super().__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # 编码器 - GRU
        self.encoder_rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # VAE 潜在空间
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, input_dim * window_size)
        )
        
        # 预测头 - 使用LayerNorm避免BatchNorm的单样本问题
        # 移除Tanh限制，支持预测任意范围的Alpha收益率（研报对齐：端到端监督学习）
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1)
            # 移除 nn.Tanh()：允许输出任意范围的值，适配中性化后的收益率标签
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GRU, nn.LSTM)):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param)
                    else:
                        nn.init.zeros_(param)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def encode(self, x):
        """编码器"""
        _, hidden = self.encoder_rnn(x)
        h = hidden[-1]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, -10, 2)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z, window_size, input_dim):
        """解码器"""
        x_flat = self.decoder(z)
        x_recon = x_flat.view(-1, window_size, input_dim)
        return x_recon
    
    def predict(self, z):
        """预测头"""
        return self.predictor(z)
    
    def forward(self, x):
        """前向传播"""
        batch_size, window_size, input_dim = x.shape
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, window_size, input_dim)
        y_pred = self.predict(z)
        return x_recon, y_pred.squeeze(), mu, logvar, z


@register_model('vae')
@register_model('VAE')
class VAEModel(PyTorchModel):
    """
    VAE (Variational Autoencoder) 时序预测模型
    
    结合了自编码器的特征提取能力和变分推断的正则化优势，
    特别适合用于因子提取和异常检测。
    
    Example:
        model = VAEModel(
            d_feat=20,
            hidden_dim=128,
            latent_dim=16,
            window_size=40,
            n_epochs=50,
            alpha_recon=0.1,
            beta_kl=0.001,
            gamma_pred=1.0
        )
        model.fit(train_loader, valid_loader)
        predictions, latent_features = model.predict(test_loader, return_latent=True)
    """
    
    def __init__(
        self,
        d_feat: int = 20,
        hidden_dim: int = 128,
        latent_dim: int = 16,
        window_size: int = 40,
        dropout: float = 0.3,
        alpha_recon: float = 0.1,
        beta_kl: float = 0.001,
        gamma_pred: float = 1.0,
        **kwargs
    ):
        """
        Args:
            d_feat: 特征维度
            hidden_dim: 隐藏层大小
            latent_dim: 潜在空间维度
            window_size: 时间窗口大小
            dropout: Dropout 概率
            alpha_recon: 重构损失权重
            beta_kl: KL散度损失权重
            gamma_pred: 预测损失权重
        """
        super().__init__(**kwargs)
        
        self.d_feat = d_feat
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.window_size = window_size
        self.dropout_rate = dropout
        self.alpha_recon = alpha_recon
        self.beta_kl = beta_kl
        self.gamma_pred = gamma_pred
        
        # 创建模型
        self.model = VAENet(
            d_feat, hidden_dim, latent_dim, window_size, dropout
        ).to(self.device)
        
        # 创建优化器
        self.optimizer = self._get_optimizer()
        
        self.logger.info(f"VAE 模型参数:")
        self.logger.info(f"  特征维度: {d_feat}")
        self.logger.info(f"  隐藏维度: {hidden_dim}")
        self.logger.info(f"  潜在维度: {latent_dim}")
        self.logger.info(f"  窗口大小: {window_size}")
        self.logger.info(f"  损失权重 - 重构: {alpha_recon}, KL: {beta_kl}, 预测: {gamma_pred}")
    
    def _compute_loss(self, x_recon, x_true, y_pred, y_true, mu, logvar):
        """计算 VAE 损失函数"""
        # 重构损失
        recon_loss = nn.functional.mse_loss(x_recon, x_true, reduction='mean')
        
        # KL散度损失
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 预测损失
        if y_pred.dim() == 0:
            y_pred = y_pred.unsqueeze(0)
        if y_true.dim() == 2:
            y_true = y_true.squeeze()
        pred_loss = nn.functional.mse_loss(y_pred, y_true, reduction='mean')
        
        # 总损失
        total_loss = (
            self.alpha_recon * recon_loss + 
            self.beta_kl * kl_loss + 
            self.gamma_pred * pred_loss
        )
        
        return total_loss, recon_loss, kl_loss, pred_loss
    
    def _train_epoch(self, data_loader):
        """训练一个 epoch (重写以支持VAE损失)"""
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        for batch_x, batch_y in tqdm(data_loader, desc="训练", leave=False):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            x_recon, y_pred, mu, logvar, z = self.model(batch_x)
            
            # 计算损失
            loss, _, _, _ = self._compute_loss(x_recon, batch_x, y_pred, batch_y, mu, logvar)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0
    
    def _valid_epoch(self, data_loader):
        """验证一个 epoch (重写以支持VAE损失)"""
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                x_recon, y_pred, mu, logvar, z = self.model(batch_x)
                loss, _, _, _ = self._compute_loss(x_recon, batch_x, y_pred, batch_y, mu, logvar)
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0
    
    def fit(self, train_loader, valid_loader=None, save_path: Optional[str] = None):
        """训练模型"""
        self.logger.info("开始训练 VAE 模型...")
        
        # 🆕 创建学习率调度器
        self.scheduler = self._get_scheduler()
        
        best_valid_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.n_epochs):
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            if valid_loader is not None:
                valid_loss = self._valid_epoch(valid_loader)
                self.valid_losses.append(valid_loss)
                
                self.logger.info(
                    f"Epoch {epoch+1}/{self.n_epochs} - "
                    f"Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}"
                )
                
                # 🆕 调用学习率调度器
                self._step_scheduler(valid_loss)
                
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    self.best_score = -valid_loss
                    self.best_epoch = epoch
                    patience_counter = 0
                    
                    if save_path:
                        self.save_model(save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stop:
                        self.logger.info(f"早停触发 at epoch {epoch+1}")
                        break
            else:
                self.logger.info(f"Epoch {epoch+1}/{self.n_epochs} - Train Loss: {train_loss:.6f}")
                self._step_scheduler(train_loss)
                self.best_epoch = epoch
        
        self.fitted = True
        if valid_loader is not None:
            self.logger.info(f"训练完成! 最佳 epoch: {self.best_epoch+1}")
        else:
            self.logger.info(f"训练完成! (无验证集，共 {self.n_epochs} epochs)")
    
    def predict(self, test_loader, return_numpy: bool = True, return_latent: bool = False):
        """
        预测（🆕 精简版：return_latent=False 时复用基类实现）
        
        Args:
            test_loader: 测试数据加载器
            return_numpy: 是否返回 numpy 数组
            return_latent: 是否同时返回潜在特征
            
        Returns:
            如果 return_latent=False: 预测结果
            如果 return_latent=True: (预测结果, 潜在特征)
        """
        # 🆕 方案 A：不需要潜变量时，直接复用基类实现
        if not return_latent:
            return super().predict(test_loader, return_numpy)
        
        # 需要潜变量时，使用自定义逻辑
        if not self.fitted:
            raise ValueError("模型未训练，请先调用 fit()")
        
        self.model.eval()
        predictions = []
        latent_features = []
        
        with torch.no_grad():
            for batch_data in test_loader:
                batch_x, _, _, _ = self._parse_batch_data(batch_data)
                batch_x = batch_x.to(self.device)
                _, y_pred, _, _, z = self.model(batch_x)
                predictions.append(y_pred.cpu())
                latent_features.append(z.cpu())
        
        # 空处理
        if len(predictions) == 0:
            import numpy as np
            empty = np.array([]) if return_numpy else torch.tensor([])
            return empty, empty
        
        predictions = torch.cat(predictions, dim=0)
        latent_features = torch.cat(latent_features, dim=0)
        
        if return_numpy:
            return predictions.numpy(), latent_features.numpy()
        return predictions, latent_features
    
    def _forward_for_predict(self, x, adj=None, idx=None):
        """🆕 VAE 预测前向钩子 - 仅返回 y_pred"""
        _, y_pred, _, _, _ = self.model(x)
        return y_pred
    
    def extract_latent(self, test_loader, return_numpy: bool = True):
        """
        提取潜在特征（用于因子生成）
        
        🆕 2026-01-11 修复：使用 _parse_batch_data 支持图/日级 loader
        
        Args:
            test_loader: 数据加载器（支持 (x,y), (x,y,adj,...), dict 等格式）
            return_numpy: 是否返回 numpy 数组
            
        Returns:
            潜在特征 (mu, z)
        """
        if not self.fitted:
            raise ValueError("模型未训练，请先调用 fit()")
        
        self.model.eval()
        mu_list = []
        z_list = []
        
        with torch.no_grad():
            for batch_data in test_loader:
                # 🆕 使用基类统一的 batch 解析（替代 for batch_x, _ in ...)
                batch_x, _, _, _ = self._parse_batch_data(batch_data)
                
                batch_x = batch_x.to(self.device)
                mu, logvar = self.model.encode(batch_x)
                z = self.model.reparameterize(mu, logvar)
                mu_list.append(mu.cpu())
                z_list.append(z.cpu())
        
        # 处理空输入
        if len(mu_list) == 0:
            import numpy as np
            empty = np.array([]) if return_numpy else torch.tensor([])
            return empty, empty
        
        mu_features = torch.cat(mu_list, dim=0)
        z_features = torch.cat(z_list, dim=0)
        
        if return_numpy:
            return mu_features.numpy(), z_features.numpy()
        return mu_features, z_features


if __name__ == '__main__':
    print("=" * 80)
    print("PyTorch Models 测试")
    print("=" * 80)
    
    # 测试模型创建
    print("\n1. 测试 LSTM 模型创建:")
    lstm_model = LSTMModel(
        d_feat=20,
        hidden_size=64,
        num_layers=2,
        n_epochs=10,
        batch_size=256
    )
    print(f"✅ LSTM 模型创建成功")
    
    print("\n2. 测试 GRU 模型创建:")
    gru_model = GRUModel(
        d_feat=20,
        hidden_size=64,
        num_layers=2
    )
    print(f"✅ GRU 模型创建成功")
    
    print("\n3. 测试 Transformer 模型创建:")
    transformer_model = TransformerModel(
        d_feat=20,
        d_model=64,
        nhead=4,
        num_layers=2
    )
    print(f"✅ Transformer 模型创建成功")
    
    print("\n4. 测试 VAE 模型创建:")
    vae_model = VAEModel(
        d_feat=20,
        hidden_dim=128,
        latent_dim=16,
        window_size=40,
        n_epochs=10
    )
    print(f"✅ VAE 模型创建成功")
    
    # 测试模型注册
    from model_factory import ModelRegistry
    print(f"\n已注册的模型: {ModelRegistry.list_models()}")
    
    print("\n✅ PyTorch Models 测试完成")
