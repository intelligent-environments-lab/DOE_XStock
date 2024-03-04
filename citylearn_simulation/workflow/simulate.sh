#!/bin/sh
python src/simulate.py run_work_order workflow/work_order/alameda.sh || exit 1
python src/simulate.py run_work_order workflow/work_order/travis.sh || exit 1
python src/simulate.py run_work_order workflow/work_order/chittenden.sh || exit 1
