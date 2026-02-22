alter table public.attempts
add column if not exists attempt_name text null;

update public.attempts
set attempt_name = regexp_replace(source_filename, '\.[^.]+$', '')
where attempt_name is null;
