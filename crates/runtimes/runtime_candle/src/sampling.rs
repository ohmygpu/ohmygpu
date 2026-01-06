//! Token sampling strategies

use anyhow::Result;
use candle_core::Tensor;

pub struct Sampler {
    temperature: f32,
    top_p: f32,
    #[allow(dead_code)]
    rng_seed: u64,
    rng_state: u64,
}

impl Sampler {
    pub fn new(temperature: f32, top_p: f32, seed: u64) -> Self {
        Self {
            temperature: temperature.max(0.001), // Avoid division by zero
            top_p,
            rng_seed: seed,
            rng_state: seed,
        }
    }

    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        let logits = logits.to_dtype(candle_core::DType::F32)?;
        let logits = logits.to_vec1::<f32>()?;

        // Apply temperature
        let scaled: Vec<f32> = logits
            .iter()
            .map(|&x| x / self.temperature)
            .collect();

        // Softmax
        let max_logit = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = scaled.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let probs: Vec<f32> = exp.iter().map(|&x| x / sum).collect();

        // Top-p (nucleus) sampling
        let token = if self.top_p < 1.0 {
            self.sample_top_p(&probs)
        } else {
            self.sample_multinomial(&probs)
        };

        Ok(token)
    }

    fn sample_top_p(&mut self, probs: &[f32]) -> u32 {
        // Sort by probability descending
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Find cutoff for top-p
        let mut cumsum = 0.0;
        let mut cutoff_idx = indexed.len();
        for (i, (_, p)) in indexed.iter().enumerate() {
            cumsum += p;
            if cumsum >= self.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Renormalize and sample
        let candidates = &indexed[..cutoff_idx];
        let sum: f32 = candidates.iter().map(|(_, p)| p).sum();
        let normalized: Vec<f32> = candidates.iter().map(|(_, p)| p / sum).collect();

        let r = self.random_f32();
        let mut cumsum = 0.0;
        for (i, p) in normalized.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return candidates[i].0 as u32;
            }
        }

        candidates.last().map(|(idx, _)| *idx as u32).unwrap_or(0)
    }

    fn sample_multinomial(&mut self, probs: &[f32]) -> u32 {
        let r = self.random_f32();
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return i as u32;
            }
        }
        (probs.len() - 1) as u32
    }

    fn random_f32(&mut self) -> f32 {
        // Simple xorshift64 PRNG
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;

        // Convert to f32 in [0, 1)
        (self.rng_state as f64 / u64::MAX as f64) as f32
    }
}
