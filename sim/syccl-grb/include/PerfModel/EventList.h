/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "PerfModel/Event.h"
#include "PerfModel/Type.h"
#include <queue>

namespace Simulation {

    /**
 * EventList encapsulates a number of Events along with its event time.
 */
    class EventList {
    public:

        explicit EventList(EventTime event_time) noexcept;

        [[nodiscard]] EventTime get_event_time() const noexcept;

        void add_event(Callback callback, CallbackArg callback_arg, unsigned int priority) noexcept;

        void invoke_events() noexcept;

    private:
        /// event time of the event list
        EventTime event_time;

        /// list of registered events
        std::priority_queue<Event, std::vector<Event>, EventPrioCompare> events;
    };

}  // namespace Simulation
