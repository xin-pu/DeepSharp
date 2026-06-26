using RLSharp.Core.Environments;
using RLSharp.Core.Spaces;

namespace RLSharp.Torch.Examples.CartPole
{
	/// <summary>
	///     Classic CartPole control environment implemented against the generic RLSharp Core API.
	/// </summary>
	public sealed class CartPoleEnvironment : IEnvironment<CartPoleState, CartPoleAction>
	{
		private const float Gravity               = 9.8f;
		private const float MassCart              = 1.0f;
		private const float MassPole              = 0.1f;
		private const float TotalMass             = MassCart + MassPole;
		private const float Length                = 0.5f;
		private const float PoleMassLength        = MassPole * Length;
		private const float ForceMagnitude        = 10.0f;
		private const float Tau                   = 0.02f;
		private const float ThetaThresholdRadians = 12f * MathF.PI / 180f;
		private const float XThreshold            = 2.4f;

		private readonly Random _random;
		private          int    _steps;

		public CartPoleEnvironment(int? seed = null, int maxSteps = 500)
		{
			_random      = seed is null ? new Random() : new Random(seed.Value);
			MaxSteps     = maxSteps;
			ActionSpace  = new DiscreteActionSpace<CartPoleAction>(Enum.GetValues<CartPoleAction>());
			CurrentState = CreateRandomInitialState();
		}

		public int MaxSteps { get; }

		public CartPoleState CurrentState { get; private set; }

		public string Name => "CartPole";

		public IActionSpace<CartPoleAction> ActionSpace { get; }

		public CartPoleState Reset()
		{
			_steps       = 0;
			CurrentState = CreateRandomInitialState();
			return CurrentState;
		}

		public StepResult<CartPoleState> Step(CartPoleAction action)
		{
			_steps++;

			var x        = CurrentState.Position;
			var xDot     = CurrentState.Velocity;
			var theta    = CurrentState.Angle;
			var thetaDot = CurrentState.AngularVelocity;

			var force    = action == CartPoleAction.Right ? ForceMagnitude : -ForceMagnitude;
			var cosTheta = MathF.Cos(theta);
			var sinTheta = MathF.Sin(theta);
			var temp     = (force + PoleMassLength * thetaDot * thetaDot * sinTheta) / TotalMass;
			var thetaAcc = (Gravity * sinTheta - cosTheta * temp) /
			               (Length * (4.0f / 3.0f - MassPole * cosTheta * cosTheta / TotalMass));
			var xAcc = temp - PoleMassLength * thetaAcc * cosTheta / TotalMass;

			x        += Tau * xDot;
			xDot     += Tau * xAcc;
			theta    += Tau * thetaDot;
			thetaDot += Tau * thetaAcc;

			CurrentState = new CartPoleState(x, xDot, theta, thetaDot);

			var terminal =
				x      < -XThreshold            ||
				x      > XThreshold             ||
				theta  < -ThetaThresholdRadians ||
				theta  > ThetaThresholdRadians  ||
				_steps >= MaxSteps;

			var reward = terminal && _steps < MaxSteps ? 0f : 1f;
			return new StepResult<CartPoleState>(CurrentState, reward, terminal);
		}

		private CartPoleState CreateRandomInitialState()
		{
			return new CartPoleState(
				RandomUniform(-0.05f, 0.05f),
				RandomUniform(-0.05f, 0.05f),
				RandomUniform(-0.05f, 0.05f),
				RandomUniform(-0.05f, 0.05f));
		}

		private float RandomUniform(float min, float max)
		{
			return min + (float)_random.NextDouble() * (max - min);
		}
	}
}