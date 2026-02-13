-- Enable extension for UUID generation
create extension if not exists pgcrypto;

-- Profiles
create table if not exists public.profiles (
  user_id uuid primary key references auth.users(id) on delete cascade,
  display_name text null,
  created_at timestamptz not null default now()
);

-- Tracks
create table if not exists public.tracks (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  name text not null,
  layout_direction text not null default 'unknown' check (layout_direction in ('cw','ccw','unknown')),
  layout_variant text null,
  location text null,
  created_at timestamptz not null default now()
);

create unique index if not exists tracks_user_name_layout_uidx
on public.tracks (user_id, lower(name), layout_direction, coalesce(lower(layout_variant), ''));

-- Attempts
create table if not exists public.attempts (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  track_id uuid not null references public.tracks(id) on delete cascade,
  source_filename text not null,
  storage_bucket text not null default 'fit-files',
  storage_path text not null,
  status text not null default 'uploaded' check (status in ('uploaded','processed','failed')),
  processing_error text null,
  method_name text not null,
  params_json jsonb not null default '{}'::jsonb,
  speed_threshold_kmh double precision null,
  laps_detected integer null,
  valid_laps integer null,
  curves_detected integer null,
  boundary_curves integer null,
  median_track_width_m double precision null,
  p90_track_width_m double precision null,
  best_lap_s double precision null,
  avg_lap_s double precision null,
  lap_time_cv double precision null,
  uploaded_at timestamptz not null default now(),
  processed_at timestamptz null
);

create index if not exists attempts_user_track_uploaded_idx
on public.attempts (user_id, track_id, uploaded_at desc);

create index if not exists attempts_track_uploaded_idx
on public.attempts (track_id, uploaded_at desc);

-- Attempt laps
create table if not exists public.attempt_laps (
  id bigserial primary key,
  attempt_id uuid not null references public.attempts(id) on delete cascade,
  lap integer not null,
  lap_time_s double precision null,
  avg_speed_kmh double precision null,
  max_speed_kmh double precision null,
  lap_distance_m double precision null,
  is_valid_lap boolean null,
  unique (attempt_id, lap)
);

-- Attempt curves
create table if not exists public.attempt_curves (
  id bigserial primary key,
  attempt_id uuid not null references public.attempts(id) on delete cascade,
  lap integer not null,
  curve_id integer not null,
  entry_speed_kmh double precision null,
  apex_speed_kmh double precision null,
  exit_speed_kmh double precision null,
  curve_time_s double precision null,
  time_loss_vs_best_s double precision null,
  s_start_m double precision null,
  s_end_m double precision null,
  s_apex_m double precision null,
  peak_curvature double precision null
);

create index if not exists attempt_curves_attempt_curve_lap_idx
on public.attempt_curves (attempt_id, curve_id, lap);

-- RLS
alter table public.profiles enable row level security;
alter table public.tracks enable row level security;
alter table public.attempts enable row level security;
alter table public.attempt_laps enable row level security;
alter table public.attempt_curves enable row level security;

-- Profiles policies
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'public'
      AND tablename = 'profiles'
      AND policyname = 'profiles_owner_all'
  ) THEN
    CREATE POLICY profiles_owner_all
    ON public.profiles
    FOR ALL
    USING (user_id = auth.uid())
    WITH CHECK (user_id = auth.uid());
  END IF;
END
$$;

-- Tracks policies
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'public'
      AND tablename = 'tracks'
      AND policyname = 'tracks_owner_all'
  ) THEN
    CREATE POLICY tracks_owner_all
    ON public.tracks
    FOR ALL
    USING (user_id = auth.uid())
    WITH CHECK (user_id = auth.uid());
  END IF;
END
$$;

-- Attempts policies
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'public'
      AND tablename = 'attempts'
      AND policyname = 'attempts_owner_all'
  ) THEN
    CREATE POLICY attempts_owner_all
    ON public.attempts
    FOR ALL
    USING (user_id = auth.uid())
    WITH CHECK (user_id = auth.uid());
  END IF;
END
$$;

-- Attempt laps policies via parent attempt
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'public'
      AND tablename = 'attempt_laps'
      AND policyname = 'attempt_laps_owner_all'
  ) THEN
    CREATE POLICY attempt_laps_owner_all
    ON public.attempt_laps
    FOR ALL
    USING (
      EXISTS (
        SELECT 1 FROM public.attempts a
        WHERE a.id = attempt_laps.attempt_id
          AND a.user_id = auth.uid()
      )
    )
    WITH CHECK (
      EXISTS (
        SELECT 1 FROM public.attempts a
        WHERE a.id = attempt_laps.attempt_id
          AND a.user_id = auth.uid()
      )
    );
  END IF;
END
$$;

-- Attempt curves policies via parent attempt
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'public'
      AND tablename = 'attempt_curves'
      AND policyname = 'attempt_curves_owner_all'
  ) THEN
    CREATE POLICY attempt_curves_owner_all
    ON public.attempt_curves
    FOR ALL
    USING (
      EXISTS (
        SELECT 1 FROM public.attempts a
        WHERE a.id = attempt_curves.attempt_id
          AND a.user_id = auth.uid()
      )
    )
    WITH CHECK (
      EXISTS (
        SELECT 1 FROM public.attempts a
        WHERE a.id = attempt_curves.attempt_id
          AND a.user_id = auth.uid()
      )
    );
  END IF;
END
$$;

-- Storage bucket and policies
insert into storage.buckets (id, name, public)
values ('fit-files', 'fit-files', false)
on conflict (id) do nothing;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'storage'
      AND tablename = 'objects'
      AND policyname = 'fit_files_owner_select'
  ) THEN
    CREATE POLICY fit_files_owner_select
    ON storage.objects
    FOR SELECT
    USING (
      bucket_id = 'fit-files'
      AND auth.uid()::text = split_part(name, '/', 1)
    );
  END IF;
END
$$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'storage'
      AND tablename = 'objects'
      AND policyname = 'fit_files_owner_insert'
  ) THEN
    CREATE POLICY fit_files_owner_insert
    ON storage.objects
    FOR INSERT
    WITH CHECK (
      bucket_id = 'fit-files'
      AND auth.uid()::text = split_part(name, '/', 1)
    );
  END IF;
END
$$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'storage'
      AND tablename = 'objects'
      AND policyname = 'fit_files_owner_update'
  ) THEN
    CREATE POLICY fit_files_owner_update
    ON storage.objects
    FOR UPDATE
    USING (
      bucket_id = 'fit-files'
      AND auth.uid()::text = split_part(name, '/', 1)
    )
    WITH CHECK (
      bucket_id = 'fit-files'
      AND auth.uid()::text = split_part(name, '/', 1)
    );
  END IF;
END
$$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'storage'
      AND tablename = 'objects'
      AND policyname = 'fit_files_owner_delete'
  ) THEN
    CREATE POLICY fit_files_owner_delete
    ON storage.objects
    FOR DELETE
    USING (
      bucket_id = 'fit-files'
      AND auth.uid()::text = split_part(name, '/', 1)
    );
  END IF;
END
$$;
