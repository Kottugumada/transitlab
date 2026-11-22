"""initial schema — all §7 tables

Revision ID: 0001
Revises:
Create Date: 2026-11-21
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "koi_catalogue",
        sa.Column("kepoi_name", sa.String(32), primary_key=True),
        sa.Column("kepid", sa.BigInteger(), nullable=True),
        sa.Column("kepler_name", sa.String(64), nullable=True),
        sa.Column("disposition", sa.String(32), nullable=True),
        sa.Column("pdisposition", sa.String(32), nullable=True),
        sa.Column("koi_score", sa.Double(), nullable=True),
        sa.Column("period", sa.Double(), nullable=True),
        sa.Column("time0", sa.Double(), nullable=True),
        sa.Column("duration", sa.Double(), nullable=True),
        sa.Column("depth", sa.Double(), nullable=True),
        sa.Column("ror", sa.Double(), nullable=True),
        sa.Column("impact", sa.Double(), nullable=True),
        sa.Column("model_snr", sa.Double(), nullable=True),
        sa.Column("odd_even_sig", sa.Double(), nullable=True),
        sa.Column("steff", sa.Double(), nullable=True),
        sa.Column("srad", sa.Double(), nullable=True),
        sa.Column("kepmag", sa.Double(), nullable=True),
        sa.Column("features", JSONB(), nullable=True),
        sa.Column("synced_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_koi_catalogue_kepid", "koi_catalogue", ["kepid"])
    op.create_index("ix_koi_catalogue_disposition", "koi_catalogue", ["disposition"])

    op.create_table(
        "ps_planets",
        sa.Column("pl_name", sa.String(128), primary_key=True),
        sa.Column("hostname", sa.String(128), nullable=True),
        sa.Column("period", sa.Double(), nullable=True),
        sa.Column("tranmid", sa.Double(), nullable=True),
        sa.Column("trandur", sa.Double(), nullable=True),
        sa.Column("trandep", sa.Double(), nullable=True),
        sa.Column("rade", sa.Double(), nullable=True),
        sa.Column("sy_vmag", sa.Double(), nullable=True),
        sa.Column("ra", sa.Double(), nullable=True),
        sa.Column("dec", sa.Double(), nullable=True),
        sa.Column("synced_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_ps_planets_hostname", "ps_planets", ["hostname"])

    op.create_table(
        "hunt_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("n_targets", sa.Integer(), nullable=True),
        sa.Column("n_completed", sa.Integer(), nullable=True),
        sa.Column("n_detections", sa.Integer(), nullable=True),
        sa.Column("config", JSONB(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
    )

    op.create_table(
        "training_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("data_snapshot", sa.Text(), nullable=True),
        sa.Column("split_seed", sa.Integer(), nullable=True),
        sa.Column("honest_features", JSONB(), nullable=True),
        sa.Column("excluded_features", JSONB(), nullable=True),
        sa.Column("metrics", JSONB(), nullable=True),
        sa.Column("model_path", sa.Text(), nullable=True),
    )

    op.create_table(
        "targets",
        sa.Column("target_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("mission", sa.String(16), nullable=False),
        sa.Column("kepid", sa.BigInteger(), nullable=True),
        sa.Column("tic_id", sa.BigInteger(), nullable=True),
        sa.Column("label_disposition", sa.String(32), nullable=True),
        sa.Column("hunt_status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index("ix_targets_kepid", "targets", ["kepid"])
    op.create_index("ix_targets_hunt_status", "targets", ["hunt_status"])

    op.create_table(
        "detections",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("target_id", sa.Integer(), sa.ForeignKey("targets.target_id"), nullable=False),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("hunt_runs.id"), nullable=True),
        sa.Column("period", sa.Double(), nullable=True),
        sa.Column("epoch_bjd", sa.Double(), nullable=True),
        sa.Column("duration_h", sa.Double(), nullable=True),
        sa.Column("depth_ppm", sa.Double(), nullable=True),
        sa.Column("bls_power", sa.Double(), nullable=True),
        sa.Column("bls_snr", sa.Double(), nullable=True),
        sa.Column("fit_depth_ppm", sa.Double(), nullable=True),
        sa.Column("fit_depth_sigma", sa.Double(), nullable=True),
        sa.Column("odd_even_sigma", sa.Double(), nullable=True),
        sa.Column("secondary_depth_ppm", sa.Double(), nullable=True),
        sa.Column("secondary_sigma", sa.Double(), nullable=True),
        sa.Column("shape_ratio", sa.Double(), nullable=True),
        sa.Column("n_transits", sa.Integer(), nullable=True),
        sa.Column("depth_scatter", sa.Double(), nullable=True),
        sa.Column("vetting", JSONB(), nullable=True),
        sa.Column("fold_data", JSONB(), nullable=True),
        sa.Column("catalogue_model_score", sa.Double(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index("ix_detections_target_id", "detections", ["target_id"])
    op.create_index("ix_detections_bls_snr", "detections", ["bls_snr"])

    op.create_table(
        "backyard_predictions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("date_local", sa.String(10), nullable=False),
        sa.Column(
            "pl_name",
            sa.String(128),
            sa.ForeignKey("ps_planets.pl_name"),
            nullable=False,
        ),
        sa.Column("hostname", sa.String(128), nullable=True),
        sa.Column("ingress", sa.DateTime(timezone=True), nullable=True),
        sa.Column("mid_transit", sa.DateTime(timezone=True), nullable=True),
        sa.Column("egress", sa.DateTime(timezone=True), nullable=True),
        sa.Column("depth_pct", sa.Double(), nullable=True),
        sa.Column("sy_vmag", sa.Double(), nullable=True),
        sa.Column("observable_rank", sa.Integer(), nullable=True),
        sa.Column("reasons", JSONB(), nullable=True),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index(
        "ix_backyard_predictions_date_local", "backyard_predictions", ["date_local"]
    )


def downgrade() -> None:
    op.drop_table("backyard_predictions")
    op.drop_table("detections")
    op.drop_table("targets")
    op.drop_table("training_runs")
    op.drop_table("hunt_runs")
    op.drop_table("ps_planets")
    op.drop_table("koi_catalogue")
