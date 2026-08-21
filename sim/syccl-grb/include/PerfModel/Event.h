/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "PerfModel/Type.h"

namespace Simulation {

    #define EVENT_PRIO_CHUNK_ARRIVAL 110
    #define EVENT_PRIO_LINK_FREE 120
    #define EVENT_PRIO_LINK_PROCESS_TRANSMIT_QUEUE 100
    class Event {
    public:
        Event(Callback callback, CallbackArg callback_arg, unsigned int priority) noexcept;
        unsigned int priority;

        /**
   * Invoke the callback function.
   */
        void invoke_event() noexcept;

        [[nodiscard]] std::pair<Callback, CallbackArg> get_handler_arg() const noexcept;

    private:
        /// pointer to the callback function
        Callback callback;

        /// argument of the callback function
        CallbackArg callback_arg;
    };

    struct EventPrioCompare {
        bool operator()(const Event& lhs, const Event& rhs) const {
            return lhs.priority < rhs.priority;
        }
    };

}  // namespace Simulation
