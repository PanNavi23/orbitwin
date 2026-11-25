#!/bin/bash
# GMAT GUI as a parse oracle: $1 = script, $2 = run dir. Test oracle only.
set -u
G=${GMAT_ROOT:?the R2026a install}; Q=${ORACLE_DIR:?a scratch dir}
mkdir -p "$2/out" "$Q/tmp"; script=$(readlink -e "$1"); run=$(readlink -e "$2"); : "${script:?}" "${run:?}"
sed -e "s|^ROOT_PATH .*|ROOT_PATH                = $G/|" \
    -e "s|^OUTPUT_PATH .*|OUTPUT_PATH              = $run/out/|" \
    -e "s|^LOG_FILE .*|LOG_FILE                 = $run/GmatLog.txt|" \
    -e "s|^PERSONALIZATION_FILE .*|PERSONALIZATION_FILE     = $run/MyGmat.ini|" \
    -e "s|^#WRITE_PERSONALIZATION_FILE = ON|WRITE_PERSONALIZATION_FILE = OFF|" \
    -e "s|^#NO_SPLASH .*|NO_SPLASH               = TRUE|" \
    "$G/bin/gmat_startup_file.txt" > "$run/startup.txt"
ulimit -c 0
cd "$G/bin" && TMPDIR=$Q/tmp LD_LIBRARY_PATH=${COMPAT_LIBS:?a dir with libtiff.so.5}:$G/lib \
  timeout -k 5 ${3:-90} xvfb-run -a -s "-screen 0 1280x1024x24" ./GMAT-R2026a -ns -m \
  --startup_file "$run/startup.txt" --run "$script" --exit > "$run/console.txt" 2>&1
rc=$?
echo "rc=$rc" > "$run/rc.txt"
L="$run/GmatLog.txt"
# Only a run to "Mission run completed" passes; an error or a crash before it does not.
if [ ! -s "$L" ]; then v=NO-LOG
elif rg -q "Failed to build the script" "$L"; then v=REFUSED-AT-BUILD
elif rg -q "\*\*\*\* ERROR|Exception" "$L"; then v=ERROR-WHEN-RUN
elif rg -q "Mission run completed" "$L"; then v=RAN-TO-END
else v=CRASHED-BEFORE-THE-END; fi
echo "$v rc=$rc $(basename "$script")"
