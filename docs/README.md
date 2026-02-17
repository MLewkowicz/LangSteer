# LangSteer Documentation

Complete documentation for the LangSteer framework.

## Quick Links

- **[Setup Guide](../SETUP.md)** - Installation and environment setup
- **[Main README](../README.md)** - Project overview and quick start

## Guides

Step-by-step tutorials for common tasks:

- **[Experiments](guides/experiments.md)** - Running experiments and evaluations
- **[Training](guides/training.md)** - Training DP3 policies and forecasters
- **[Visualization](guides/visualization.md)** - Visualizing trajectories and camera feeds

## Reference

Detailed technical documentation:

- **[Forecasters](reference/forecasters.md)** - Forecaster models and training
- **[Rollout System](reference/rollout.md)** - Episode runner and data collection

## Migration

Guides for transitioning from old systems:

- **[Visualization Migration](migration/visualization.md)** - Migrating from old visualization scripts

## Directory Structure

```
docs/
├── README.md                    # This file
├── guides/                      # How-to guides
│   ├── experiments.md          # Running experiments
│   ├── training.md             # Training models
│   └── visualization.md        # Visualization system
├── reference/                   # Reference documentation
│   ├── forecasters.md          # Forecaster models
│   └── rollout.md              # Rollout system
└── migration/                   # Migration guides
    └── visualization.md        # Old → new visualization
```

## Contributing

When adding new documentation:

1. **Guides** - Place in `docs/guides/` for step-by-step tutorials
2. **Reference** - Place in `docs/reference/` for API/technical docs
3. **Keep it DRY** - Link to existing docs rather than duplicating
4. **Update this index** - Add new docs to the appropriate section above

## Getting Help

- Check the relevant guide or reference doc first
- See [SETUP.md](../SETUP.md) for installation issues
- Report bugs at: https://github.com/your-repo/issues
