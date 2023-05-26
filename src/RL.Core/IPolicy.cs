using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;

namespace RL.Core
{
   public interface IPolicy
    {
        public torch.Tensor Predict(torch.Tensor input);
    }
}
